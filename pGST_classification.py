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

        train_ls = []
        test_ls = []
        for i in range(self.lenth):
            if i%int(1/test_rate)==0:
                test_ls.append(i)
            else:
                train_ls.append(i)

        # train_idx = int(self.lenth*(1-test_rate))


        if split == 'train':
            self.GSTcoe = GSTcoe_all[train_ls]
            self.labels = label_all[train_ls]
        elif split == 'test':
            self.GSTcoe = GSTcoe_all[test_ls]
            self.labels = label_all[test_ls]
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
        self.mlp1 = nn.Linear(in_features=self.nodeNum, out_features=midnum, bias=True)
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
    parser.add_argument("--dataset", type=str, default='DD', help="name of dataset")
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--epochs", type=int, default= 10000)
    parser.add_argument("--batchsize", type=int, default= 4, help="batch size of dataset")
    parser.add_argument('--workers',default=1,type=int, metavar='N')
    parser.add_argument("--numScales", type=int, default= 5, help="size of filter bank")
    parser.add_argument("--numLayers", type=int, default= 5, help="layers of GST")
    parser.add_argument("--waytogetphi", type=str, default= "generate", help="store or generate")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,8"
    num_gst_coe = computeNcoe(args.numScales, args.numLayers)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.waytogetphi == "generate" :
        data = dgl.data.TUDataset('PROTEINS')  ## 'ENZYMES' 'PROTEINS' 'COLLAB' 'DD'
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

        # dataloader = GraphDataLoader(data, batch_size=1, shuffle=True)
        label_all = np.zeros(len(data))
        bar = Bar('>>>', fill='>', max=len(data))
        for k,(g, labels) in enumerate(data):
            
            A = np.zeros((g.num_nodes(),g.num_nodes()))
            for i in range(g.num_edges()):
                A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1

            GSTmodel = np_GST.DiffusionScattering(args.numScales, args.numLayers, A)        
            node_attr = np.expand_dims(np.expand_dims(g.ndata['node_attr'].numpy(), axis=0), axis=0)
            co_GST = GSTmodel.computeTransform(node_attr)
            if k == 0:
                num_coe = np.shape(co_GST)[2]
                GSTcoe_all = np.zeros((len(data),1,num_coe))
                print(np.shape(GSTcoe_all))
            GSTcoe_all[k] = co_GST[0]
            label_all[k] = int(labels.item())
            bar.next()
        bar.finish()
        print(label_all)
        label_all = label_all*2-1
        print(label_all)

    ## --------------------------------建立训练和测试数据集------------------------------------------------------------------
    if args.waytogetphi == "store" :
        GSTcoe_all = np.load('/DATA7_DB7/data/gjliu/dataset/PROTEINS/allphi_PROTEINS.npy')
        label_all = np.load('/DATA7_DB7/data/gjliu/dataset/PROTEINS/alllabel_PROTEINS.npy')
    num_coe = np.shape(GSTcoe_all)[2]

    GSTdataset_train = GST_coef_dataset(GSTcoe_all, label_all, 'train')
    GSTdataset_test = GST_coef_dataset(GSTcoe_all, label_all, 'test')
    print('len(GSTdataset_train):',len(GSTdataset_train))
    print('len(GSTdataset_test):',len(GSTdataset_test))
    train_loader = torch.utils.data.DataLoader(GSTdataset_train, batch_size=args.batchsize,
                   shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(GSTdataset_test, batch_size=args.batchsize,
                   shuffle=False, pin_memory=True)
    model = Perceptron(input_dim = num_coe)
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(60), gamma=0.5)
    
    st = time.time()

    ##------------------------------------- 训练阶段-----------------------------------------------
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for epoch in range(10000):
        st_epoch = time.time()
        # print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        # print('-' * 10)

        loss_train = []
        for i, (input, target) in enumerate(train_loader):


            input_var = input.to(device).float()
            target_var =target.to(device).int()
            # print(np.shape(input_var))
            # print(np.shape(target_var))

            # SVM 前向传播
            scores = model(input_var).squeeze(1).squeeze(1)
            loss = loss_func(scores, target_var, "svm")

            # MLP 向前传播


            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())
                # for name,param in model.named_parameters():
                #     print(name, param)

        scheduler.step()
        if epoch % 20 == 0:
            # 计算分类的准确率
            acc = valid(test_loader,model)
            acc_train = valid(train_loader,model)
            print("loss=", np.mean(loss_train),"acc=", acc,"acctrain=", acc_train) #loss.detach().cpu().numpy()

            # print('zantin')
            # input()        
            print('Epoch {}/{}'.format(epoch, args.epochs - 1))
            
            bt_epoch = time.time()
            print('epoch time:',bt_epoch-st_epoch,'  total time:', bt_epoch - st)
            print('-' * 10)
        bar.next()
    bar.finish()


