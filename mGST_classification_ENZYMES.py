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
from dgl.dataloading import GraphDataLoader
from sklearn import preprocessing 
from progress.bar import Bar
from sklearn.ensemble import RandomForestClassifier
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
    def __init__(self, GSTcoe_all, label_all, split='train', test_rate=0.2, if_normalize=True):

        self.lenth = len(label_all)
        self.normalize = if_normalize

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
            self.num_1 = np.sum(label_all)
        elif split == 'test':
            self.GSTcoe = GSTcoe_all[test_ls]
            self.labels = label_all[test_ls]
            self.num_1 = np.sum(label_all)
        else:
            raise RuntimeError('Invalid split')

        if self.normalize:
            phis_mean = np.expand_dims(np.mean(self.GSTcoe,axis=2), axis=2)
            phis_std = np.expand_dims(np.std(self.GSTcoe, axis=2), axis=2)
            self.GSTcoe = (self.GSTcoe - phis_mean)/phis_std
            self.GSTcoe[np.isnan(self.GSTcoe)] = 0 # phis_std may be zero, remove invalid values here   


    def __getitem__(self, index):
        return self.GSTcoe[index,:,:], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

class MLPs(nn.Module):
    def __init__(self, class_num=2, midnum = 256, input_dim=None):
        super(MLPs, self).__init__()
        self.input_dim = input_dim
        self.mlp1 = nn.Linear(in_features=self.input_dim, out_features=midnum, bias=True)
        # self.dropout1 = nn.Dropout(0.5)
        self.mlp2 = nn.Linear(in_features=midnum, out_features=midnum, bias=True)
        self.mlp3 = nn.Linear(in_features=midnum, out_features=class_num, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.relu(x)
        # x = self.dropout1(x)
        # x = self.mlp2(x)
        # x = self.relu(x)
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
    # labels[labels>0]=1
    # labels[labels<=0]=0    
    # print(labels)
    pred_scores=np.array(pred_scores)
    # print(pred_scores)
    # pred_scores[pred_scores>0]=1
    # pred_scores[pred_scores<=0]=0
    # print(pred_scores)
    acc= accuracy_score(labels, pred_scores)
    return acc

def valid2(test_loader,model):
    pred_scores=[]
    labels=[]
    for j, (input, target) in enumerate(test_loader):
        input = input.reshape(np.shape(input)[0],1,np.shape(input)[1]*np.shape(input)[2])
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
    pred_scores=np.array(pred_scores)

    acc= accuracy_score(labels, pred_scores)
    return acc

def GSTgenerator_Pyramid(data, variation, args):
    bar = Bar('>>>', fill='>', max=len(data))
    label_all = np.zeros(len(data))

    num_coe = computeNcoe(args.numScales, args.numLayers)
    GSTcoe_all_p = np.zeros((len(data),args.feature_choosen,num_coe,variation))

    for k,(g, labels) in enumerate(data):
        label_all[k] = int(labels.item())
        A = np.zeros((g.num_nodes(),g.num_nodes()))
        for i in range(g.num_edges()):
            A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1
        node_attr = np.expand_dims(np.transpose(g.ndata['node_attr'].numpy()[:,0:args.feature_choosen],(1,0)), axis=0)
        # node_attr = np.expand_dims(np.expand_dims(g.ndata['node_attr'].numpy(), axis=0), axis=0)
        A_multiscale = A.copy()
        node_attr_multiscale = node_attr.copy()

        for ptimes in range(variation):

            GSTmodel = np_GST.DiffusionScattering(args.numScales, args.numLayers, A_multiscale)        
            # print('node_attr_multiscale',np.shape(node_attr_multiscale))
            co_GST = GSTmodel.computeTransform(node_attr_multiscale)

            if args.gstcoe_norml == 'MS':
                for c in range(np.shape(co_GST)[1]):
                    co_GST[0,c] = Z_ScoreNormalization(co_GST[0,c])
            if args.gstcoe_norml == '01':
                for c in range(np.shape(co_GST)[1]):
                    co_GST[0,c] = m01_Normalization(co_GST[0,c])

            num_coe = np.shape(co_GST)[2]

            GSTcoe_all_p[k,:,:,ptimes] = co_GST[0]
            # GSTcoe_all[k] = co_GST[0]
            
            node_sampled, node_sampled_c = downsample(A_multiscale, method='umax', laplacian_basic = 'normalized', trick = 0)
            Laplacian = laplacian(A_multiscale, laplacian_basic = 'normalized')
            R_Laplacian = Kron_Reduction(Laplacian, node_sampled, node_sampled_c, mode=args.reduction_mode)
            sampled_lenth = np.shape(R_Laplacian)[0]
            A_multiscale = np.multiply(R_Laplacian,(np.eye(sampled_lenth)-np.ones((sampled_lenth,sampled_lenth))))
            A_multiscale[np.isnan(A_multiscale)] = 0
            # sampled_lenth = np.shape(R_Laplacian)[0]
            # A_multiscale = np.multiply(R_Laplacian,(np.eye(sampled_lenth)-np.ones((sampled_lenth,sampled_lenth))))
            # A_multiscale[np.isnan(A_multiscale)] = 0
##########################################
            # A_multiscale = A_multiscale[np.squeeze(node_sampled, axis=0),:]
            # A_multiscale = A_multiscale[:,np.squeeze(node_sampled, axis=0)]
###########################################
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
    GSTcoe_all_p = np.zeros((len(data),args.feature_choosen,num_coe, int(math.pow(2, variation)-1)))

    for k,(g, labels) in enumerate(data):
        # (g, labels)= data[k+758]
        label_all[k] = int(labels.item())
        A = np.zeros((g.num_nodes(),g.num_nodes()))
        for i in range(g.num_edges()):
            A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1
        node_attr = np.expand_dims(np.transpose(g.ndata['node_attr'].numpy()[:,0:args.feature_choosen],(1,0)), axis=0)
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
## 一些比较好的数据记录----------------------------
#batchsize-16\\dataset-pro\\lr=0.1\\step_size=int(100), gamma=0.5---72.5
#batchsize-4\\dataset-pro\\lr=0.02\\step_size=int(60), gamma=0.5---73.9
##--------------------------------------------------------------------------------


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description="GST configuration")
    parser.add_argument("--data_path", type=str, default='/DATA7_DB7/data/gjliu/dataset/PROTEINS', help="path of dataset")
    parser.add_argument("--dataset", type=str, default='ENZYMES', help="name of dataset")
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--epochs", type=int, default= 10000)
    parser.add_argument("--test_times", type=int, default= 5)
    parser.add_argument("--batchsize", type=int, default= 4, help="batch size of dataset")
    parser.add_argument('--workers',default=1,type=int, metavar='N')
    parser.add_argument("--numScales", type=int, default= 5, help="size of filter bank")
    parser.add_argument("--numLayers", type=int, default= 5, help="layers of GST")
    parser.add_argument("--midnum", type=int, default= 64, help="nums of hidden layers")
    parser.add_argument("--waytogetphi", type=str, default= "generate", help="store or generate")
    parser.add_argument("--log_path", type=str, default= "/DB/rhome/gjliu/workspace/graph_scattering/log", help="store or generate")

    parser.add_argument("--variation", type=int, default= 2, help="kinds of GST input")
    parser.add_argument("--feature_choosen", type=int, default= 18, help="feature channels to be choosen")
    parser.add_argument("--reduction_mode", type=str, default= 'simple', help="way of graph structure reduction")
    parser.add_argument("--tips", type=str, default= 'original')
    parser.add_argument("--gstcoe_norml", type=str, default= 'MS')
    parser.add_argument("--sample", type=str, default= 'umax', help='umax,umin,way to sample')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    num_gst_coe = computeNcoe(args.numScales, args.numLayers)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(filename=args.log_path + "/"+args.dataset+"/Log_"+str(args.variation)+".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.waytogetphi == "generate" :
        data = dgl.data.TUDataset(args.dataset)  ## 'ENZYMES' 'PROTEINS' 'COLLAB' 'DD'
        print('Number of categories:', data.num_labels)
        print('Number of categories:', data.num_labels)
        ## -------------------------------------生成GST系数------------------------------------------------------------------------

        # dataloader = GraphDataLoader(data, batch_size=1, shuffle=True)
        # label_all = np.zeros(len(data))
        # bar = Bar('>>>', fill='>', max=len(data))
        # for k,(g, labels) in enumerate(data):
            
        #     A = np.zeros((g.num_nodes(),g.num_nodes()))
        #     for i in range(g.num_edges()):
        #         A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1

        #     GSTmodel = np_GST.DiffusionScattering(args.numScales, args.numLayers, A)

        #     node_attr = np.expand_dims(np.transpose(g.ndata['node_attr'].numpy()[:,0:18],(1,0)), axis=0)
        #     # node_attr = np.expand_dims(np.expand_dims(g.ndata['node_attr'].numpy(), axis=0), axis=0)
        #     co_GST = GSTmodel.computeTransform(node_attr)
        #     if k == 0:
        #         num_coe = np.shape(co_GST)[2]
        #         num_feature = np.shape(co_GST)[1]
        #         GSTcoe_all = np.zeros((len(data),num_feature,num_coe))
        #         print(np.shape(GSTcoe_all))
        #     GSTcoe_all[k] = co_GST[0]
        #     label_all[k] = int(labels.item())
        #     bar.next()
        # bar.finish()

        GSTcoe_all, label_all = GST_Pyramid_complete(data, variation=args.variation, args = args)
        GSTcoe_all[np.isnan(GSTcoe_all)] = 0

    ## --------------------------------建立训练和测试数据集------------------------------------------------------------------
    if args.waytogetphi == "store" :
        GSTcoe_all = np.load('/DATA7_DB7/data/gjliu/dataset/PROTEINS/allphi_PROTEINS.npy')
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
    GSTcoe_val = GSTcoe_all[val_ls]
    labels_val = label_all[val_ls]

    GSTcoe_train = GSTcoe_train.reshape(np.shape(GSTcoe_train)[0],1,np.shape(GSTcoe_train)[1]*np.shape(GSTcoe_train)[2])
    GSTcoe_test = GSTcoe_test.reshape(np.shape(GSTcoe_test)[0],1,np.shape(GSTcoe_test)[1]*np.shape(GSTcoe_test)[2])
    # GSTcoe_val = GSTcoe_val.reshape(np.shape(GSTcoe_val)[0],1,np.shape(GSTcoe_val)[1]*np.shape(GSTcoe_val)[2])

    print('size of train set:', len(labels_train))
    print('size of test set:', len(labels_test))
    # print('size of val set:', len(labels_val))

    # best_acc = 0
    best_forest_size = 2000
    for best_forest_size in [10000,30000,100000]:
        logging.info('Start testing')
        test_acc_lst = []
        for i in range(args.test_times):
            classifier = RandomForestClassifier(n_estimators=best_forest_size)
            start_time = time.time()
            classifier.fit(np.squeeze(GSTcoe_train), labels_train)
            train_acc = classifier.score(np.squeeze(GSTcoe_train), labels_train)
            test_acc = classifier.score(np.squeeze(GSTcoe_test), labels_test)
            test_acc_lst.append(test_acc)

            logging.info('n_estimators %d train accuracy %f : ' % (best_forest_size, train_acc))
            logging.info('n_estimators %d test accuracy %f : ' % (best_forest_size, test_acc))
            print('Time used for RF classification:', time.time() - start_time)
            logging.info('Time used for RF classification: %d ' % (time.time() - start_time))
            print('=' * 20)
        test_acc_mean = np.mean(test_acc_lst)
        logging.info('n_estimators %d mean test accuracy %f : ' % (best_forest_size, test_acc_mean))

    ##------------------------------------- 训练阶段-----------------------------------------------
    # train_loader = torch.utils.data.DataLoader(GSTdataset_train, batch_size=args.batchsize,
    #                shuffle=True, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(GSTdataset_test, batch_size=args.batchsize,
    #                shuffle=False, pin_memory=True)
    # model = MLPs(class_num=data.num_labels, input_dim = num_coe*num_feature, midnum = args.midnum)
    # model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    # model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = SGD(model.parameters(), lr=0.01, weight_decay=5e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(100), gamma=0.5)
    # bar = Bar('>>>', fill='>', max=len(train_loader))
    # for epoch in range(args.epochs):
    #     st_epoch = time.time()
    #     # print('Epoch {}/{}'.format(epoch, args.epochs - 1))
    #     # print('-' * 10)

    #     loss_train = []
    #     for i, (input, target) in enumerate(train_loader):

    #         input = input.reshape(np.shape(input)[0],1,np.shape(input)[1]*np.shape(input)[2])

    #         input_var = input.to(device).float()
    #         target_var =target.to(device).long()
    #         # print(np.shape(input_var))
    #         # print(np.shape(target_var))

    #         # SVM 前向传播
    #         # scores = model(input_var).squeeze(1).squeeze(1)
    #         # loss = loss_func(scores, target_var, "svm")

    #         # MLP 向前传播
    #         scores = np.squeeze(model(input_var))

    #         loss = criterion(scores, target_var)

    #         # 反向传播
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         loss_train.append(loss.item())
    #             # for name,param in model.named_parameters():
    #             #     print(name, param)

    #     scheduler.step()
    #     if epoch % 60 == 0:
    #         # 计算分类的准确率
    #         acc = valid2(test_loader,model)
    #         acc_train = valid2(train_loader,model)
    #         print("loss=", np.mean(loss_train),"acc=", acc,"acctrain=", acc_train) #loss.detach().cpu().numpy()

    #         # print('zantin')
    #         # input()        
    #         print('Epoch {}/{}'.format(epoch, args.epochs - 1))
            
    #         bt_epoch = time.time()
    #         print('epoch time:',bt_epoch-st_epoch,'  total time:', bt_epoch - st)
    #         print('-' * 10)



    #     bar.next()
    # bar.finish()


