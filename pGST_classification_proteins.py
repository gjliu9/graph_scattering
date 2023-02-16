#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# system package
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
# import dgl.data.TUDataset as TUDataset
# our package
from Modules.STGST_torch_s2 import STGSTModule
import Modules.graphScattering as np_GST
from graph_multiscale import *

class GST_coef_dataset(Dataset):
    def __init__(self, GSTcoe_all, label_all, split='train', test_rate=0.2, val_rate=0.14):

        self.lenth = len(label_all)

        # if self.normalize:
        #     phis_mean = np.mean(phis[train_idx],axis=0)
        #     phis_std = np.std(phis[train_idx], axis=0)
        #     phis = (phis - phis_mean) / phis_std
        #     phis[np.isnan(phis)] = 0 # phis_std may be zero, remove invalid values here
        #     phis[np.isinf(phis)] = 0

        train_ls = []
        test_ls = []
        val_ls = []
        for i in range(self.lenth):
            if i%int(1/test_rate)==0:
            #     val_ls.append(i)
            # elif i%int(1/test_rate)==0:
                test_ls.append(i)
            else:
                train_ls.append(i)

        # train_idx = int(self.lenth*(1-test_rate))


        if split == 'train':
            self.GSTcoe = GSTcoe_all[train_ls]
            self.labels = label_all[train_ls]
            self.num_1 = np.sum(self.labels)
        elif split == 'test':
            self.GSTcoe = GSTcoe_all[test_ls]
            self.labels = label_all[test_ls]
            self.num_1 = np.sum(self.labels)
        elif split == 'val':
            self.GSTcoe = GSTcoe_all[val_ls]
            self.labels = label_all[val_ls]
            self.num_1 = np.sum(self.labels)
        else:
            raise RuntimeError('Invalid split')


    def __getitem__(self, index):
        return self.GSTcoe[index,:,:], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

class MLPs(nn.Module):
    def __init__(self, class_num=2, midnum = 64, input_dim=None):
        super(MLPs, self).__init__()
        self.input_dim = input_dim
        self.mlp1 = nn.Linear(in_features=self.input_dim, out_features=midnum, bias=True)
        # self.dropout1 = nn.Dropout(0.5)
        self.mlp2 = nn.Linear(in_features=midnum, out_features=class_num, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.relu(x)
        # x = self.dropout1(x)
        x = self.mlp2(x)
        return x


def computeNcoe(scale,layers):
    num = 0
    for i in range(layers):
        num = num + pow(scale, i)
    return num

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.layer = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        x = self.layer(x)
        return x

def sign(x):
    x[x>=0] = 1
    x[x<0] = -1
    return x

def loss_func(scores, label, type="svm"):
    assert type=="perceptron" or type=="svm", "loss type error"
    if type == "perceptron":
        # 感知机损失函数，label取值集合为{-1, 1}
        loss = -label*scores
    else:
        # SVM损失函数，label取值集合为{-1, 1}
        loss = 1-label*scores
   
    loss[loss<=0] = 0
    return torch.sum(loss)

def pred(x):
    return sign(x)

def valid(test_loader,model):
    pred_scores=[]
    labels=[]
    for j, (input, target) in enumerate(test_loader):
        input_var = input.to(device).float()
        target_var =target.to(device).int()
        scores = model(input_var).squeeze(1).squeeze(1)
        for m in range(len(target)):
            pred_scores.append(scores[m].item())
            labels.append(np.float(target[m].numpy()))

    labels = np.array(labels)
    # print(labels)
    labels[labels>0]=1
    labels[labels<=0]=0    
    # print(labels)
    pred_scores=np.array(pred_scores)
    # print(pred_scores)
    pred_scores[pred_scores>0]=1
    pred_scores[pred_scores<=0]=0
    # print(pred_scores)
    acc= accuracy_score(labels, pred_scores)
    return acc

def valid2(test_loader,model):
    pred_scores=[]
    labels=[]
    for j, (input, target) in enumerate(test_loader):
        input_var = input.to(device).float()
        target_var =target.to(device).long()
        scores = np.squeeze(model(input_var))
#         print(scores)
        for m in range(len(target)):
#             print((scores[m].detach().cpu().numpy().tolist()).index(max(scores[m])))
            pred_scores.append((scores[m].detach().cpu().numpy().tolist()).index(max(scores[m])))
            labels.append(np.float(target[m].numpy()))
    # print('label',labels)
    # print('pred_scores',pred_scores)
    labels = np.array(labels)

    labels[labels>0]=1
    labels[labels<=0]=0    

    pred_scores=np.array(pred_scores)

    acc= accuracy_score(labels, pred_scores)
    return acc

# def graph_stochastic_sampling(A,p):
#     # input adjacency matrix:A
#     N = np.shape(A)[0]
#     rdmx = np.random.rand(N,N)
# #     print(rdmx)
#     rdmx = (rdmx<p)
# #     print(rdmx)
#     sampled_A = A * rdmx
# #     print(sampled_A)
#     return sampled_A

# def Kron_Reduction(laplacion, sampled_list, sampled_list_c):

#     sampled_list = np.squeeze(sampled_list)
#     sampled_list_c = np.squeeze(sampled_list_c)
#     L_11 = laplacion[sampled_list,:]
#     L_11 = L_11[:,sampled_list]

#     L_10 = laplacion[sampled_list,:]
#     L_10 = L_10[:,sampled_list_c]

#     L_01 = laplacion[sampled_list_c,:]
#     L_01 = L_01[:,sampled_list]

#     L_00 = laplacion[sampled_list_c,:]
#     L_00 = L_00[:,sampled_list_c]
#     L_00 = np.linalg.inv(L_00)

#     L_reduct = L_11 - L_10 @ L_00 @ L_01

#     return L_reduct

def GSTgenerator_Pyramid(data, variation, args):
    bar = Bar('>>>', fill='>', max=len(data))
    label_all = np.zeros(len(data))

    num_coe = computeNcoe(args.numScales, args.numLayers)
    GSTcoe_all_p = np.zeros((len(data),1,num_coe,variation))

    for k,(g, labels) in enumerate(data):
        # (g, labels)= data[k+758]
        label_all[k] = int(labels.item())
        A = np.zeros((g.num_nodes(),g.num_nodes()))
        for i in range(g.num_edges()):
            A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1
        node_attr = np.expand_dims(np.expand_dims(g.ndata['node_attr'].numpy(), axis=0), axis=0)
        A_multiscale = A.copy()
        node_attr_multiscale = node_attr.copy()

        for ptimes in range(variation):

            GSTmodel = np_GST.DiffusionScattering(args.numScales, args.numLayers, A_multiscale)        
            # print('node_attr_multiscale',np.shape(node_attr_multiscale))
            co_GST = GSTmodel.computeTransform(node_attr_multiscale)


            if args.gstcoe_norml == 'MS':
                co_GST = Z_ScoreNormalization(co_GST)
            if args.gstcoe_norml == '01':
                co_GST = m01_Normalization(co_GST)

            num_coe = np.shape(co_GST)[2]

            GSTcoe_all_p[k,:,:,ptimes] = co_GST[0]
            # GSTcoe_all[k] = co_GST[0]
            
            node_sampled, node_sampled_c = downsample(A_multiscale, method='umax', laplacian_basic = 'normalized', trick = 0)
            Laplacian = laplacian(A_multiscale, laplacian_basic = 'normalized')
            R_Laplacian = Kron_Reduction(Laplacian, node_sampled, node_sampled_c, mode=args.reduction_mode)
            sampled_lenth = np.shape(R_Laplacian)[0]
            A_multiscale = np.multiply(R_Laplacian,(np.eye(sampled_lenth)-np.ones((sampled_lenth,sampled_lenth))))
            A_multiscale[np.isnan(A_multiscale)] = 0
            node_attr_multiscale = node_attr_multiscale[:,:,np.squeeze(node_sampled, axis=0)]
            node_attr_multiscale[np.isnan(node_attr_multiscale)] = 0

        bar.next()
    bar.finish()
    # np.save('/DATA7_DB7/data/gjliu/dataset/'+'PROTEINS'+'/allphi_p_'+'PROTEINS'+'.npy',GSTcoe_all_p)
    GSTcoe_all = GSTcoe_all_p.reshape(np.shape(GSTcoe_all_p)[0],np.shape(GSTcoe_all_p)[1],np.shape(GSTcoe_all_p)[2]*np.shape(GSTcoe_all_p)[3])
    return  GSTcoe_all, label_all #

def downsample(A, method='umax', laplacian_basic = 'unnormalized', trick = 1):
    ###########################################
    ## 根据邻接矩阵和特定采样方式挑选出被采样的节点
    ###########################################
    # print('打印L：\n{}'.format(L))
    # print('打印特征值a：\n{}'.format(eigenvalue))
    # print('打印特征向量b：\n{}'.format(eigenvector))
    if method=='umax':
        L = laplacian(A, laplacian_basic)
        eigenvalue, eigenvector = np.linalg.eig(L)
        maxindex = np.argmax(eigenvalue)
        max_eigenvector = eigenvector[:,maxindex]
        # location = max_eigenvector >= 0
        location = np.where(max_eigenvector >= 0)
        location_c = np.where(max_eigenvector < 0)
    if method=='umin':
        L = laplacian(A, laplacian_basic)
        eigenvalue, eigenvector = np.linalg.eig(L)
        eigenvalue[eigenvalue<1e-10] = max(eigenvalue) + 1
        minindex = np.argmin(eigenvalue)
        min_eigenvector = eigenvector[:,minindex]
        # print('打印特征向量b：\n{}'.format(min_eigenvector))
        if trick == 0:
            location = np.where(min_eigenvector >= 0)
            location_c = np.where(min_eigenvector < 0)
        elif trick == 1:
            location = np.where(min_eigenvector <= 0)
            location_c = np.where(min_eigenvector > 0)
    return location, location_c

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
    np.save('/DATA7_DB7/data/gjliu/dataset/'+args.dataset+'/allphi_p_'+args.dataset+'_'+str(args.variation)+'_'+args.sample+'_'+args.gstcoe_norml+'.npy',GSTcoe_all_p)
    GSTcoe_all = GSTcoe_all_p.reshape(np.shape(GSTcoe_all_p)[0],np.shape(GSTcoe_all_p)[1],np.shape(GSTcoe_all_p)[2]*np.shape(GSTcoe_all_p)[3])
    return  GSTcoe_all, label_all #

## 一些比较好的数据记录----------------------------
#batchsize-16\\dataset-pro\\lr=0.1\\step_size=int(100), gamma=0.5---72.5
#batchsize-4\\dataset-pro\\lr=0.02\\step_size=int(60), gamma=0.5---73.9
#batchsize-4\\dataset-pro\\lr=0.01\\step_size=int(60), gamma=0.5 midlayer=128---74.4
##--------------------------------------------------------------------------------


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
    parser.add_argument("--numScales", type=int, default= 5, help="size of filter bank")
    parser.add_argument("--numLayers", type=int, default= 5, help="layers of GST")
    parser.add_argument("--reduction_mode", type=str, default= 'simple', help="way of graph structure reduction")
    parser.add_argument("--midnum", type=int, default= 64, help="nums of hidden layers")
    parser.add_argument("--waytogetphi", type=str, default= "load", help="store or generate")
    parser.add_argument("--log_path", type=str, default= "/DB/rhome/gjliu/workspace/graph_scattering/log/proteins", help="store or generate")
    parser.add_argument("--variation", type=int, default= 3, help="kinds of GST input")
    parser.add_argument("--tips", type=str, default= 'original')
    parser.add_argument("--gstcoe_norml", type=str, default= 'MS')
    parser.add_argument("--sample", type=str, default= 'umin', help='umax,umin,way to sample')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,8"
    num_gst_coe = computeNcoe(args.numScales, args.numLayers)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(filename=args.log_path + "/Log_"+str(args.variation)+".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.waytogetphi == "generate" :
        data = dgl.data.TUDataset(args.dataset)  ## 'ENZYMES' 'PROTEINS' 'COLLAB' 'DD'
        # g, label = data[1024]
        print('Number of categories:', data.num_labels)
        # print('Number of num_nodes:', g.num_nodes())
        # print('Number of num_edges:', g.num_edges())
        # print('ndata:', g.ndata['node_attr'])
        # print('edata:', g.edata)
        # print('edges:',g.edges())
        # print('length',len(data))
        GSTcoe_all, label_all = GST_Pyramid_complete(data, variation=args.variation, args = args)
        GSTcoe_all[np.isnan(GSTcoe_all)] = 0

    ## --------------------------------建立训练和测试数据集------------------------------------------------------------------
    if args.waytogetphi == "load" :
        GSTcoe_all = np.load('/DATA7_DB7/data/gjliu/dataset/'+args.dataset+'/allphi_p_'+args.dataset+'_'+str(args.variation)+'_'+args.sample+'_'+args.gstcoe_norml+'.npy')
        label_all = np.load('/DATA7_DB7/data/gjliu/dataset/PROTEINS/alllabel_PROTEINS.npy')
        num_coe = np.shape(GSTcoe_all)[2]

    

    train_ls = []
    test_ls = []
    val_ls = []
    kk = 0
    for i in range(len(label_all)):
        if i%4==0:  ### int(1/0.2)
            test_ls.append(i)
        else :
            train_ls.append(i)

    GSTcoe_train = GSTcoe_all[train_ls]
    labels_train = label_all[train_ls]

    GSTcoe_test = GSTcoe_all[test_ls]
    labels_test = label_all[test_ls]

    print('labels_test',labels_test)

    GSTcoe_val = GSTcoe_all[val_ls]
    labels_val = label_all[val_ls]

    GSTcoe_train = GSTcoe_train.reshape(np.shape(GSTcoe_train)[0],1,np.shape(GSTcoe_train)[1]*np.shape(GSTcoe_train)[2])
    GSTcoe_test = GSTcoe_test.reshape(np.shape(GSTcoe_test)[0],1,np.shape(GSTcoe_test)[1]*np.shape(GSTcoe_test)[2])
    # GSTcoe_val = GSTcoe_val.reshape(np.shape(GSTcoe_val)[0],1,np.shape(GSTcoe_val)[1]*np.shape(GSTcoe_val)[2])

    print('size of train set:', len(labels_train))
    print('size of test set:', len(labels_test))

    best_forest_size = 6000

    # Testing
    logging.info('Start testing')
    for k in [20000,60000,150000]:
        test_acc_lst = []
        for i in range(args.test_times):
        
            # classifier = GradientBoostingClassifier(n_estimators=k)
            classifier = RandomForestClassifier(n_estimators=k)

            start_time = time.time()
            classifier.fit(np.squeeze(GSTcoe_train), labels_train)
            train_acc = classifier.score(np.squeeze(GSTcoe_train), labels_train)
            test_acc = classifier.score(np.squeeze(GSTcoe_test), labels_test)
            test_acc_lst.append(test_acc)

            best_forest_size = k
            logging.info('n_estimators %d train accuracy %f : ' % (best_forest_size, train_acc))
            logging.info('best_estimators %d Classification accuracy %f : ' % (best_forest_size, test_acc))
            logging.info('Time used for RF classification: %d ' % (time.time() - start_time))
            print('=' * 20)
        test_acc_mean = np.mean(test_acc_lst)
        logging.info('n_estimators %d mean test accuracy %f : ' % (best_forest_size, test_acc_mean))




