#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# system package
import argparse
import logging
import math
import os
import pickle
import sys
import time
from os import listdir
from os.path import join

import dgl
import dgl.data
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
from dgl.dataloading import GraphDataLoader
from progress.bar import Bar
from sklearn import preprocessing, tree
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from torch.optim import SGD
from torch.utils.data import Dataset
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

import Modules.graphScattering as np_GST
from graph_multiscale import *
# import dgl.data.TUDataset as TUDataset
# our package
from Modules.STGST_torch_s2 import STGSTModule


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
        self.dropout1 = nn.Dropout(0.5)
        self.mlp2 = nn.Linear(in_features=midnum, out_features=class_num, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.mlp2(x)
        return x

class MLPs2(nn.Module):
    def __init__(self, class_num=2, midnum = 64, input_dim=None):
        super(MLPs2, self).__init__()
        self.input_dim = input_dim
        self.mlp1 = nn.Linear(in_features=self.input_dim, out_features=class_num, bias=True)
        # self.dropout1 = nn.Dropout(0.5)
        # self.mlp2 = nn.Linear(in_features=midnum, out_features=class_num, bias=True)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.mlp1(x)
        # x = self.relu(x)
        # x = self.dropout1(x)
        # x = self.mlp2(x)
        return x

class MLPs4(nn.Module):
    def __init__(self, class_num=2, midnum = 64, input_dim=None):
        super(MLPs4, self).__init__()
        self.input_dim = input_dim
        self.mlp1 = nn.Linear(in_features=self.input_dim, out_features=midnum, bias=True)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.mlp2 = nn.Linear(in_features=midnum, out_features=midnum, bias=True)
        self.mlp3 = nn.Linear(in_features=midnum, out_features=class_num, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.mlp2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.mlp3(x)
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
    # np.save('/DATA7_DB7/data/gjliu/dataset/'+'PROTEINS'+'/allphi_p_'+'PROTEINS'+'.npy',GSTcoe_all_p)
    GSTcoe_all = GSTcoe_all_p.reshape(np.shape(GSTcoe_all_p)[0],np.shape(GSTcoe_all_p)[1],np.shape(GSTcoe_all_p)[2]*np.shape(GSTcoe_all_p)[3])
    return  GSTcoe_all, label_all #

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):

        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def generate_GST(node_features,g,args):
    node_features = np.transpose(np.expand_dims(node_features, axis=0),(0,2,1))
    A = np.zeros((g.num_nodes(),g.num_nodes()))
    for i in range(g.num_edges()):
        A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1
    GSTmodel = np_GST.DiffusionScattering(args.numScales, args.numLayers, A)
    co_GST_map = GSTmodel.computeTransform_map(node_features)
    # GSTcoe_all, label_all = GST_Pyramid_complete(data, variation=args.variation, args = args)
    # GSTcoe_all[np.isnan(GSTcoe_all)] = 0
    co_GST_map = np.transpose(np.squeeze(co_GST_map, axis=0),(1,0,2))
    co_GST_map = np.reshape(co_GST_map,(np.shape(co_GST_map)[0],-1))
    return co_GST_map

def generate_GSTmap_mul(g,args):
    A = np.zeros((g.num_nodes(),g.num_nodes()))
    for i in range(g.num_edges()):
        A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1
    node_features = g.ndata['feat']
    node_attr = np.transpose(np.expand_dims(node_features, axis=0),(0,2,1))
    n_features = node_features.shape[1]
    map_all = np.zeros((args.variation,n_features,g.num_nodes(),computeNcoe(args.numScales,args.numLayers)))
    A_multiscale = A.copy()
    node_attr_multiscale = node_attr.copy()
    A_multiscale_list = []
    node_attr_list = []
    order_list = []
    A_multiscale_list.append(A_multiscale)
    node_attr_list.append(node_attr_multiscale)
    order_list.append(np.array(list(range(g.num_nodes()))))

    for ptimes in range(args.variation):
        
        for subset in range(int(math.pow(2, ptimes))):

            A_multiscale = A_multiscale_list[int(subset/2)+int(math.pow(2, ptimes-1)-1)]
            node_attr_multiscale = node_attr_list[int(subset/2)+int(math.pow(2, ptimes-1)-1)]
            order_multiscale = order_list[int(subset/2)+int(math.pow(2, ptimes-1)-1)]
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
                    order_multiscale = order_multiscale[np.squeeze(node_sampled, axis=0)]
                A_multiscale_list.append(A_multiscale)
                node_attr_list.append(node_attr_multiscale)
                order_list.append(order_multiscale)
            if A_multiscale.size>0:
                GSTmodel = np_GST.TightHann(args.numScales, args.numLayers, A_multiscale)        
                co_GST_map_now = GSTmodel.computeTransform_map(node_attr_multiscale)
                map_all[ptimes,:,order_multiscale,:] = np.transpose(np.squeeze(co_GST_map_now, axis=0),(1,0,2))


    # np.save('/DATA7_DB7/data/gjliu/dataset/'+'PROTEINS'+'/allphi_p_'+'PROTEINS'+'.npy',GSTcoe_all_p)
    np.save('/DB/data/gjliu/Cora/co_GST_map_multi_TH'+str(args.variation)+'.npy',map_all)
    print('GST save done')
    map_all = (np.transpose(map_all,(2,0,1,3))).reshape(np.shape(map_all)[2], -1)
    map_all_twoside = (np.transpose(map_all,(2,0,1,3))).reshape(np.shape(map_all)[2], -1)
    map_all_twoside = map_all_twoside

    return  map_all #

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description="GST configuration")
    parser.add_argument("--data_path", type=str, default='/DATA7_DB7/data/gjliu/dataset/PROTEINS', help="path of dataset")
    parser.add_argument("--dataset", type=str, default='CORA', help="name of dataset")
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--epochs", type=int, default= 10000)
    parser.add_argument("--test_times", type=int, default= 5)
    parser.add_argument("--batchsize", type=int, default= 4, help="batch size of dataset")
    parser.add_argument('--workers',default=1,type=int, metavar='N')
    parser.add_argument("--numScales", type=int, default= 3, help="size of filter bank")
    parser.add_argument("--numLayers", type=int, default= 5, help="layers of GST")
    parser.add_argument("--reduction_mode", type=str, default= 'simple', help="way of graph structure reduction")
    parser.add_argument("--midnum", type=int, default= 64, help="nums of hidden layers")
    parser.add_argument("--waytogetphi", type=str, default= "load", help="store or generate")
    parser.add_argument("--log_path", type=str, default= "/DB/rhome/gjliu/workspace/graph_scattering/log", help="store or generate")
    parser.add_argument("--variation", type=int, default= 3, help="kinds of GST input")
    parser.add_argument("--tips", type=str, default= 'original')
    parser.add_argument("--gstcoe_norml", type=str, default= 'MS')
    parser.add_argument("--sample", type=str, default= 'umin', help='umax,umin,way to sample')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    num_gst_coe = computeNcoe(args.numScales, args.numLayers)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count())
    logging.basicConfig(filename=args.log_path + "/"+args.dataset+"/Log_"+str(args.variation)+".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    dataset = dgl.data.CoraGraphDataset()  ## 'ENZYMES' 'PROTEINS' 'COLLAB' 'DD'
    g = dataset[0]
    num_class = dataset.num_classes

    node_features = g.ndata['feat']
    node_labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    valid_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)
    print('num of labels',n_labels)
    
    if args.waytogetphi=='generate':
        print('start generate GST')
        start_time = time.time()
        # co_GST_map = generate_GST(node_features,g,args)
        co_GST_map = generate_GSTmap_mul(g,args)
        logging.info('Time used for GST: %d ' % (time.time() - start_time))
        print('GST done')

    else:
        print('start load GST')
        co_GST_map = np.load('/DB/data/gjliu/Cora/co_GST_map_multi_TH'+str(args.variation)+'.npy')
        co_GST_map = (np.transpose(co_GST_map,(2,0,1,3))).reshape(np.shape(co_GST_map)[2], -1)
        # co_GST_map = np.load('/DB/data/gjliu/Cora/co_GST_map_PCA9999.npy')
    print('co_GST_map',np.shape(co_GST_map))

    # ## PCA
    # # pca = PCA(n_components=0.95)
    # # pca.fit(co_GST_map)
    # # co_GST_map = pca.transform(co_GST_map)
    # 

    # co_GST_map = np.load('/DB/data/gjliu/Cora/co_GST_map.npy')
    # print(np.shape(co_GST_map))


    ## 分类模型
    # co_GST_map = m01_Normalization(co_GST_map)
    co_GST_map = torch.Tensor(co_GST_map)
    # g.ndata['feat'] = co_GST_map
    # co_GST_map = node_features
    # model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
    # opt = torch.optim.Adam(model.parameters(), lr=0.01)
    n_F = np.shape(co_GST_map)[1]
    model = MLPs(class_num=n_labels, midnum = 300, input_dim=n_F)
    # model = SAGE(in_feats=n_F, hid_feats=100, out_feats=n_labels)

    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=int(20), gamma=0.2)
    co_GST_map_var = co_GST_map.to(device)
    node_labels = node_labels.to(device)
    ## 训练阶段
    for epoch in range(200):
        model.train()
        # 使用所有节点(全图)进行前向传播计算
        
        logits = model(co_GST_map_var)
        # 计算损失值
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        # 计算验证集的准确度
        acc = evaluate(model, g, co_GST_map_var, node_labels, test_mask)
        acc_train = evaluate(model, g, co_GST_map_var, node_labels, train_mask)
        # 进行反向传播计算
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        # print('loss:',loss.item())
        # print('epoch:',epoch,'acc:',acc,'acctrain:',acc_train)
        logging.info('loss %f : ' % (loss.item()))
        logging.info('epoch: %d acc: %f acctrain: %f ' % (epoch, acc, acc_train))

