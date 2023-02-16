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
import dgl
import dgl.data
from dgl.dataloading import GraphDataLoader
from progress.bar import Bar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve,auc,accuracy_score
import pickle
import math
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


class GST_coef_dataset(Dataset):
    def __init__(self, GSTcoe_all, label_all, split='train', test_rate=0.2):

        self.lenth = len(label_all)

        # if self.normalize:
        #     phis_mean = np.mean(phis[train_idx],axis=0)
        #     phis_std = np.std(phis[train_idx], axis=0)
        #     phis = (phis - phis_mean) / phis_std
        #     phis[np.isnan(phis)] = 0 # phis_std may be zero, remove invalid values here
        #     phis[np.isinf(phis)] = 0

        train_idx = int(self.lenth*(1-test_rate))

        if split == 'train':
            self.GSTcoe = GSTcoe_all[0:train_idx]
            self.labels = label_all[0:train_idx]
        elif split == 'test':
            self.GSTcoe = GSTcoe_all[(train_idx):]
            self.labels = label_all[(train_idx):]
        else:
            raise RuntimeError('Invalid split')

    def __getitem__(self, index):
        return self.GSTcoe[index,:,:], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

class MLPs(nn.Module):
    def __init__(self, class_num=2, midnum = 128, nodeNum=None):
        super(MLPs, self).__init__()
        self.nodeNum = nodeNum
        self.mlp1 = nn.Linear(in_features=self.nodeNum*63, out_features=midnum, bias=True)
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


## 一些比较好的数据记录----------------------------
#batchsize-16\\dataset-pro\\lr=0.1\\step_size=int(100), gamma=0.5---72.5
#batchsize-4\\dataset-pro\\lr=0.02\\step_size=int(60), gamma=0.5---73.9
##--------------------------------------------------------------------------------


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description="GST configuration")
    parser.add_argument("--datadir", type=str, default='/DATA7_DB7/data/gjliu/dataset', help="path of dataset")
    parser.add_argument("--dataset", type=str, default='COLLAB', help="name of dataset,'ENZYMES' 'PROTEINS' 'COLLAB' 'DD'")
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--epochs", type=int, default= 10000)
    parser.add_argument("--batchsize", type=int, default= 4, help="batch size of dataset")
    parser.add_argument('--workers',default=1,type=int, metavar='N')
    parser.add_argument("--numScales", type=int, default= 5, help="size of filter bank")
    parser.add_argument("--numLayers", type=int, default= 5, help="layers of GST")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    num_gst_coe = computeNcoe(args.numScales, args.numLayers)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = dgl.data.TUDataset(args.dataset)  ## 'ENZYMES' 'PROTEINS' 'COLLAB' 'DD'
    # data = dgl.data.GINDataset('DD', self_loop=True)


    # g, label = data[1024]
    print('Number of categories:', data.num_labels)
    # print('Number of num_nodes:', g.num_nodes())
    # print('Number of num_edges:', g.num_edges())
    # print('ndata:', g.ndata['node_attr'])
    # print('edata:', g.edata)
    # print('edges:',g.edges())
    # print('length',len(data))

    ## -------------------------------------生成GST系数------------------------------------------------------------------------
    ## DD数据集没有node attribute，设置node attribute为1或者节点的度
    ################################################################################
    ################################################################################
    ################################################################################

    # dataloader = GraphDataLoader(data, batch_size=1, shuffle=True)
    label_all = np.zeros(len(data))
    bar = Bar('>>>', fill='>', max=len(data))
    for k,(g, labels) in enumerate(data):
        
        A = np.zeros((g.num_nodes(),g.num_nodes()))
        for i in range(g.num_edges()):
            A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1

        GSTmodel = np_GST.DiffusionScattering(args.numScales, args.numLayers, A)       

        ## 生成node attribute
        fake_node_attr = np.ones(g.num_nodes())
        node_attr = np.expand_dims(np.expand_dims(fake_node_attr, axis=0), axis=0)
        # node_attr = np.expand_dims(np.expand_dims(g.ndata['node_attr'].numpy(), axis=0), axis=0)
        ## ENZYMES数据集生成node attribute
        # node_attr = np.expand_dims(g.ndata['node_attr'].permute(1,0).numpy(), axis=0)

        co_GST = GSTmodel.computeTransform(node_attr)
        if k == 0:
            num_coe = np.shape(co_GST)[2]
            GSTcoe_all = np.zeros((len(data),1,num_coe))
            print(np.shape(GSTcoe_all))
        GSTcoe_all[k] = co_GST[0]
        label_all[k] = int(labels.item())
        bar.next()
    bar.finish()
    np.save('/DATA7_DB7/data/gjliu/dataset/'+args.dataset+'/allphi_'+args.dataset+'.npy',GSTcoe_all)
    np.save('/DATA7_DB7/data/gjliu/dataset/'+args.dataset+'/alllabel_'+args.dataset+'.npy',label_all)