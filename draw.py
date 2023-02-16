# encoding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from pylab import *    
from os import listdir
import os
from os.path import join
import numpy as np
import time
import torch
import math
import argparse
import logging
import dgl
import dgl.data
from sklearn import preprocessing  
from dgl.dataloading import GraphDataLoader
from progress.bar import Bar
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import roc_auc_score, roc_curve,auc,accuracy_score
import pickle
import math
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
from torch.optim import SGD
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# import dgl.data.TUDataset as TUDataset
# our package
from Modules.STGST_torch_s2 import STGSTModule
import Modules.graphScattering as np_GST
from graph_multiscale import *                             #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
def draw_exp_curve():
    names = ['1', '2', '3', '4', '5']
    x = range(len(names))
    y_proteins = [76.7,75.9,78.8,77.4,77.3]
    y_DD=[74.16,75.18,77.22,78.5,76]
    y_collab=[78.4,79.2,77.8,77.2,77]
    y_enzymes=[59.7,54.9,55.6,45.6,50.6]
    #plt.plot(x, y, 'ro-')
    #plt.plot(x, y1, 'bo-')
    plt.xlim(0, 6)  # 限定横轴的范围
    plt.ylim(40, 90)  # 限定纵轴的范围
    plt.figure(dpi=300) ## ,figsize=(32,40)
    # plt.plot(x, y_proteins, marker='*', ms=10,label='Proteins')
    plt.plot(x, y_DD, marker='*', ms=10,label='DD')
    # plt.plot(x, y_collab, marker='*', ms=10,label='Collab')
    # plt.plot(x, y_enzymes, marker='*', ms=10,label='Enzymes')
    # plt.legend()  # 让图例生效
    plt.xticks(x, names) #, rotation=45
    plt.margins(0.1)
    plt.subplots_adjust(bottom=0.1)
    plt.xlabel("Number of scale layers") #X轴标签
    plt.ylabel("Classification accuracy on DD") #Y轴标签
    # plt.title("The effect of different layers") #标题

    plt.grid(linestyle='-.')

    plt.show()
    plt.savefig('/DB/rhome/gjliu/workspace/graph_scattering/test_graph/exptresults.jpg')

def GST_Pyramid_complete(data, variation, args):
    bar = Bar('>>>', fill='>', max=len(data))
    label_all = np.zeros(len(data))

    num_coe = computeNcoe(args.numScales, args.numLayers)
    GSTcoe_all_p = np.zeros((len(data),1,num_coe, int(math.pow(2, variation)-1)))

    for k,(g, labels) in enumerate(data):
        # (g, labels)= data[k+758]
        label_all[k] = int(labels.item())
        A = np.zeros((g.num_nodes(),g.num_nodes()))
        for i in range(g.num_edges()):
            A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1
        node_attr = np.expand_dims(np.expand_dims(g.ndata['node_attr'].numpy(), axis=0), axis=0)
        A_multiscale = A.copy()
        node_attr_multiscale = node_attr.copy()
        A_multiscale_list = []
        node_attr_list = []
        A_multiscale_list.append(A_multiscale)
        node_attr_list.append(node_attr_multiscale)

        for ptimes in range(variation):
            
            for subset in range(int(math.pow(2, ptimes))):

                A_multiscale = A_multiscale_list[int(subset/2)+int(math.pow(2, ptimes-1)-1)]
                node_attr_multiscale = node_attr_list[int(subset/2)+int(math.pow(2, ptimes-1)-1)]
                if ptimes>0:
                    if A_multiscale.size==0:
                        node_sampled, node_sampled_c = np.array([]), np.array([])
                    else:
                        node_sampled, node_sampled_c = downsample(A_multiscale, method=args.sample, laplacian_basic = 'normalized', trick = subset%2)
                        Laplacian = laplacian(A_multiscale, laplacian_basic = 'normalized')
                        R_Laplacian = Kron_Reduction(Laplacian, node_sampled, node_sampled_c, mode=args.reduction_mode)
                        sampled_lenth = np.shape(R_Laplacian)[0]
                        A_multiscale = np.multiply(R_Laplacian,(np.eye(sampled_lenth)-np.ones((sampled_lenth,sampled_lenth))))
                        A_multiscale[np.isnan(A_multiscale)] = 0
                        node_attr_multiscale = node_attr_multiscale[:,:,np.squeeze(node_sampled, axis=0)]
                        node_attr_multiscale[np.isnan(node_attr_multiscale)] = 0
                    A_multiscale_list.append(A_multiscale)
                    node_attr_list.append(node_attr_multiscale)

                if A_multiscale.size==0:
                    co_GST = np.zeros(np.shape(co_GST))
                else:
                    GSTmodel = np_GST.DiffusionScattering(args.numScales, args.numLayers, A_multiscale)        
                    co_GST = GSTmodel.computeTransform(node_attr_multiscale)

                if args.gstcoe_norml == 'MS':
                    co_GST = Z_ScoreNormalization(co_GST)
                if args.gstcoe_norml == '01':
                    co_GST = m01_Normalization(co_GST)

                GSTcoe_all_p[k,:,:,int(math.pow(2, ptimes))-1+subset] = co_GST[0]

        bar.next()
    bar.finish()
    np.save('/GPFS/data/gjliu/PROTEINS/allphi_p_'+'PROTEINS'+'_'+str(args.variation)+'_'+args.sample+'_'+args.gstcoe_norml+'.npy',GSTcoe_all_p)
    GSTcoe_all = GSTcoe_all_p.reshape(np.shape(GSTcoe_all_p)[0],np.shape(GSTcoe_all_p)[1],np.shape(GSTcoe_all_p)[2]*np.shape(GSTcoe_all_p)[3])
    return  GSTcoe_all, label_all #

def draw_hitmap():
    print('start load')
    GSTcoe_all = np.load('/GPFS/data/gjliu/PROTEINS/allphi_p_'+'PROTEINS'+'_'
    +str(args.variation)+'_'+args.sample+'_'+args.gstcoe_norml+'.npy')
    for i in range(4):
        print(np.shape(GSTcoe_all[i,0,:,:]))
if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description="GST configuration")
    parser.add_argument("--data_path", type=str, default='/DATA7_DB7/data/gjliu/dataset/PROTEINS', help="path of dataset")
    parser.add_argument("--dataset", type=str, default='PROTEINS', help="name of dataset")
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--epochs", type=int, default= 10000)
    parser.add_argument("--test_times", type=int, default= 5)
    parser.add_argument("--batchsize", type=int, default= 4, help="batch size of dataset")
    parser.add_argument('--workers',default=1,type=int, metavar='N')
    parser.add_argument("--numScales", type=int, default= 3, help="size of filter bank")
    parser.add_argument("--numLayers", type=int, default= 3, help="layers of GST")
    parser.add_argument("--reduction_mode", type=str, default= 'simple', help="way of graph structure reduction")
    parser.add_argument("--midnum", type=int, default= 64, help="nums of hidden layers")
    parser.add_argument("--waytogetphi", type=str, default= "load", help="store or generate")
    parser.add_argument("--log_path", type=str, default= "/DB/rhome/gjliu/workspace/graph_scattering/log/proteins", help="store or generate")
    parser.add_argument("--variation", type=int, default= 5, help="kinds of GST input")
    parser.add_argument("--tips", type=str, default= 'original')
    parser.add_argument("--gstcoe_norml", type=str, default= 'none')
    parser.add_argument("--sample", type=str, default= 'umin', help='umax,umin,way to sample')
    
    args = parser.parse_args()
    data = dgl.data.TUDataset(args.dataset)  ## 'ENZYMES' 'PROTEINS' 'COLLAB' 'DD'
    # g, label = data[1024]
    print('Number of categories:', data.num_labels)
    # print('Number of num_nodes:', g.num_nodes())
    # print('Number of num_edges:', g.num_edges())
    # print('ndata:', g.ndata['node_attr'])
    # print('edata:', g.edata)
    # print('edges:',g.edges())
    # print('length',len(data))
    # GSTcoe_all, label_all = GST_Pyramid_complete(data, variation=args.variation, args = args)
    draw_hitmap()