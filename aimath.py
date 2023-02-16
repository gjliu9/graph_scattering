#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# system package
from os import listdir
import os
from os.path import join
from networkx.algorithms.centrality import eigenvector
import numpy as np
import time
from numpy.core.numeric import Inf
import torch
import math
import argparse
import csv
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
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io as scio
from matplotlib import colors
# import dgl.data.TUDataset as TUDataset
# our package
from Modules.STGST_torch_s2 import STGSTModule
import Modules.graphScattering as np_GST


def draw_graph(A,node_attr):
    ############################
    ## 根据邻接矩阵画出图结构
    ############################
    G=nx.Graph()
    num_nodes = np.shape(A)[0]
    G.add_nodes_from(np.array(range(num_nodes)).tolist())
    for i in range(num_nodes):
        for j in range(num_nodes):
            if A[i][j]>0:
                G.add_edge(i, j)

    # color = [node_attr[i] for i in range(num_nodes)]
    nx.draw(G, 
        with_labels=False, #这个选项让节点有名称
        edge_color='b', # b stands for blue! 
        pos=nx.spring_layout(G), # 这个是选项选择点的排列方式，具体可以用 help(nx.drawing.layout) 查看
                                    # 主要有spring_layout  (default), random_layout, circle_layout, shell_layout   
                                    # 这里是环形排布，还有随机排列等其他方式  
        node_color= 'r' , # r = red  //////color
        node_size=30, # 节点大小
        width=3, # 边的宽度
       )
    
    return G

def Z_ScoreNormalization(x):
    x = (x - x.mean()) / x.std()
    x[np.isinf(x)] = 0
    x[np.isnan(x)] = 0
    return x

def m01_Normalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x[np.isinf(x)] = 0
    x[np.isnan(x)] = 0
    return x

def downsample(A, method='umax', laplacian_basic = 'unnormalized', trick = 0):
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
        else:
            location = np.where(min_eigenvector <= 0)
            location_c = np.where(min_eigenvector > 0)
    return location, location_c

def laplacian(adj_matrix, laplacian_basic = 'unnormalized'):
    # laplacian矩阵
    if laplacian_basic == 'unnormalized':
    # 先求度矩阵
        R = np.sum(adj_matrix, axis=1)
        degreeMatrix = np.diag(R)
        laplacian = degreeMatrix - adj_matrix

    # 对称归一化的laplacian矩阵
    elif laplacian_basic == 'normalized':
        R = np.sum(adj_matrix, axis=1)
        R_sqrt = 1/np.sqrt(R)
        D_sqrt = np.diag(R_sqrt)
        D_sqrt[np.isinf(D_sqrt)] = 0
        D_sqrt[np.isnan(D_sqrt)] = 0
        I = np.eye(adj_matrix.shape[0])
        laplacian = I - D_sqrt @ adj_matrix @ D_sqrt
        laplacian[np.isinf(laplacian)] = 0
        laplacian[np.isnan(laplacian)] = 0
    return laplacian

def lazy_diffusion(adj_matrix):
    # laplacian矩阵
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1/np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    D_sqrt[np.isinf(D_sqrt)] = 0
    I = np.eye(adj_matrix.shape[0])
    A = D_sqrt @ adj_matrix @ D_sqrt
    T = 0.5*(I+A)
    return T

def Kron_Reduction(laplacion, sampled_list, sampled_list_c,mode='kron'):


    sampled_list = np.squeeze(sampled_list, axis=0)
    sampled_list_c = np.squeeze(sampled_list_c, axis=0)

    L_11 = laplacion[sampled_list,:]
    L_11 = L_11[:,sampled_list]

    if mode == 'simple':
        L_reduct = L_11
        return L_reduct

    if mode == 'kron':
        L_10 = laplacion[sampled_list,:]
        L_10 = L_10[:,sampled_list_c]

        L_01 = laplacion[sampled_list_c,:]
        # print('L01',np.shape(L_01))
        L_01 = L_01[:,sampled_list]

        L_00 = laplacion[sampled_list_c,:]
        L_00 = L_00[:,sampled_list_c]
        L_00[np.isinf(L_00)] = 0
        L_00[np.isnan(L_00)] = 0
        # print(np.shape(L_00))
        L_00 = np.linalg.pinv(L_00)
        # L_00 = np.linalg.inv(L_00)
        L_00[np.isinf(L_00)] = 0
        L_00[np.isnan(L_00)] = 0

        L_reduct = L_11 - L_10 @ L_00 @ L_01

    return L_reduct

def eigen_matrix(matrix):
    A = np.array([[3,-1],[-1,3]])
    print('打印A：\n{}'.format(A))
    a, b = np.linalg.eig(A)
    print('打印特征值a：\n{}'.format(a))
    print('打印特征向量b：\n{}'.format(b))
    return a, b

def draw_coords(coords):
    plt.xlim(xmax=-60,xmin=-130)
    plt.ylim(ymax=50,ymin=24)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(coords[:,0],coords[:,1],'ro')
    # plt.savefig('/DB/rhome/gjliu/workspace/graph_scattering/test_graph/US_map.jpg')

def draw_two_coords(coords, coords_c):
    plt.xlim(xmax=-60,xmin=-130)
    plt.ylim(ymax=50,ymin=24)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(coords[:,0],coords[:,1],'ro')
    plt.plot(coords_c[:,0],coords_c[:,1],'bo')
    # plt.savefig('/DB/rhome/gjliu/workspace/graph_scattering/test_graph/US_map.jpg')

def diffusion_wavelets_polynomial(j=1,x=0):
    atom = 1-0.5*x
    if j == 0:
        f = 1-atom
    else:
        f = np.power(atom, np.power(2,j-1))-np.power(atom, np.power(2,j))
    return f

def draw_diffusion_filter_spectrum(j=1, adjacancy=None):
    L = laplacian(adjacancy, laplacian_basic = 'normalized')
    eigenvalue, eigenvector = np.linalg.eig(L)
    Spectrum = np.zeros(len(eigenvalue))
    for i in range(len(Spectrum)):
        Spectrum[i] = diffusion_wavelets_polynomial(j,eigenvalue[i])
    plt.xlabel("eigenvalue")
    plt.ylabel("Spectrum of filter")
    plt.plot(eigenvalue,Spectrum,'ro')

def compute_lazy_diffusion(adj=None):
    d = np.sum(adj, axis = 1)
    N = np.shape(adj)[0]
    killIndices = np.nonzero(d < 1e-6)[0] # Nodes with zero
        # degree or negative degree (there shouldn't be any since (i) the
        # graph is connected -all nonzero degrees- and (ii) all edge
        # weights are supposed to be positive)
    dReady = d.copy()
    dReady[killIndices] = 1.
    # Apply sqrt and invert without fear of getting nans or stuff
    dSqrtInv = 1./np.sqrt(dReady)
    # Put back zeros in those numbers that had been failing
    dSqrtInv[killIndices] = 0.
    # Inverse diagonal squareroot matrix
    DsqrtInv = np.diag(dSqrtInv)
    # Normalized adjacency
    A = DsqrtInv.dot(adj.dot(DsqrtInv))
    # Lazy diffusion
    T = 1/2*(np.eye(N) + A)
    return T

def draw_diffusion_filter_spectrum_real(j=1, adjacancy=None, Map=None, signal = None):
    # input_map = Map[0,0,:,0]
    # output_map = Map[0,0,:,j]
    
    L = laplacian(adjacancy, laplacian_basic = 'normalized')

    T = compute_lazy_diffusion(adj=adjacancy)

    eigenvalue, eigenvector = np.linalg.eig(L)
    Spectrum = np.zeros(len(eigenvalue))

    filter_bank = np_GST.diffusionWavelets(3, T)
    input_map = signal
    output_map = filter_bank[j-1,:,:] @ input_map
    input_spectrum = np.transpose(eigenvector,(1,0)) @ input_map
    output_spectrum = np.transpose(eigenvector,(1,0)) @ output_map
    for i in range(len(Spectrum)):
        Spectrum[i] = output_spectrum[i]/input_spectrum[i]
    Spectrum[np.isinf(Spectrum)] = 0
    plt.xlabel("eigenvalue")
    plt.ylabel("Spectrum of filter")
    plt.plot(eigenvalue,Spectrum,'ro')

def draw_input_relu_spectrum(j=1, adjacancy=None, Map=None, signal = None):
    L = laplacian(adjacancy, laplacian_basic = 'normalized')
    T = compute_lazy_diffusion(adj=adjacancy)
    eigenvalue, eigenvector = np.linalg.eig(L)
    Spectrum = np.zeros(len(eigenvalue))
    filter_bank = np_GST.diffusionWavelets(3, T)
    input_map = signal
    output_map = filter_bank[j-1,:,:] @ input_map
    input_spectrum = np.transpose(eigenvector,(1,0)) @ input_map
    output_spectrum = np.transpose(eigenvector,(1,0)) @ output_map
    for i in range(len(Spectrum)):
        Spectrum[i] = output_spectrum[i]/input_spectrum[i]
    Spectrum[np.isinf(Spectrum)] = 0
    plt.xlabel("eigenvalue")
    plt.ylabel("Spectrum of filter")
    plt.plot(eigenvalue,output_spectrum,'ro')

def draw_input_relu_spectrum(j=1, adjacancy=None, Map=None, signal = None):
    L = laplacian(adjacancy, laplacian_basic = 'normalized')
    T = compute_lazy_diffusion(adj=adjacancy)
    eigenvalue, eigenvector = np.linalg.eig(L)
    Spectrum = np.zeros(len(eigenvalue))
    filter_bank = np_GST.diffusionWavelets(3, T)
    input_map = signal
    output_map = filter_bank[j-1,:,:] @ input_map
    input_spectrum = np.transpose(eigenvector,(1,0)) @ input_map
    output_spectrum = np.transpose(eigenvector,(1,0)) @ output_map
    for i in range(len(Spectrum)):
        Spectrum[i] = output_spectrum[i]/input_spectrum[i]
    Spectrum[np.isinf(Spectrum)] = 0

    getmap = Map[0,0,:,j]    
    output_spectrum_relu = np.transpose(eigenvector,(1,0)) @ getmap
    output_spectrum_relu[np.isinf(output_spectrum_relu)] = 0
    plt.plot(eigenvalue,output_spectrum_relu,'bo')
    plt.plot(eigenvalue,output_spectrum,'ro')
    plt.xlabel("eigenvalue")
    plt.ylabel("Spectrum of signal")
    
    

def draw_diffusion_filter_vertex(j=1, adjacancy=None, coords=None):
    plt.xlim(xmax=-60,xmin=-130)
    plt.ylim(ymax=50,ymin=24)
    L = laplacian(adjacancy, laplacian_basic = 'normalized')
    eigenvalue, eigenvector = np.linalg.eig(L)
    Spectrum = np.zeros(len(eigenvalue))
    for i in range(len(Spectrum)):
        Spectrum[i] = diffusion_wavelets_polynomial(j,eigenvalue[i])
    vertex_map = eigenvector @ Spectrum
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.plot(coords[:,0],coords[:,1],'ro')
    plt.scatter(coords[:,0],coords[:,1],c=vertex_map, cmap="OrRd")
    plt.colorbar()

def signal_generator(k=0.3, adjacancy=None, gtype="high", coords=None,D = None): ## gtype="low"//gtype="all"
    L = laplacian(adjacancy, laplacian_basic = 'normalized')
    eigenvalue, eigenvector = np.linalg.eig(L)
    order = np.argsort(eigenvalue)
    eigenvalue_rise = eigenvalue[order]
    eigenvector_rise = eigenvector[:,order]

    Spectrum = np.zeros(len(eigenvalue))
    if gtype=="high":
        for i in range(len(Spectrum)):
            if i >= (1-k)*len(Spectrum):
                Spectrum[i] = 1
    elif gtype=="low":
        for i in range(len(Spectrum)):
            if i <= k*len(Spectrum):
                Spectrum[i] = 1        
    elif gtype=="allpass":
        for i in range(len(Spectrum)):
            Spectrum[i] = 1
    
    # vertex_map = eigenvector_rise @ Spectrum
    vertex_map = (D['temperature'])[:,0]
    vertex_map = m01_Normalization(vertex_map)
    plt.figure(dpi=300)
    changecolor = colors.Normalize(vmin=0, vmax=1)
    plt.xlim(xmax=-60,xmin=-130)
    plt.ylim(ymax=50,ymin=24)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(coords[:,0],coords[:,1],c=np.abs(vertex_map), cmap="OrRd",norm=changecolor)
    plt.colorbar()
    return vertex_map

def write_csv(co_gst, mode="w+"):
    with open("/DB/rhome/gjliu/workspace/graph_scattering/test_graph/multiscale_GST_record.csv", mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["id", "category"])
        writer.writerow(co_gst.tolist())

def draw_GSTmap(map_GST, coords, J=3, L=2):
    num_map = np.shape(map_GST)[3]
    plt.figure(dpi=300,figsize=(80, 20)) ##7*(J**(L-1))

    for l in range(L):
        for j in range(J**(l)):
            plt.subplot(L,J**(L-1),j+1+l*(J**(L-1)))
            changecolor = colors.Normalize(vmin=0, vmax=1)
            plt.xlim(xmax=-60,xmin=-130)
            plt.ylim(ymax=50,ymin=24)
            plt.xlabel("x")
            plt.ylabel("y")

            plt.scatter(coords[:,0],coords[:,1],c=map_GST[0,0,:,j], cmap="OrRd",norm=changecolor)
            plt.colorbar()

def graph_stochastic_sampling(A,p):
    # input adjacency matrix:A
    N = np.shape(A)[0]
    rdmx = np.random.rand(N,N)
#     print(rdmx)
    rdmx = (rdmx<p)
#     print(rdmx)
    sampled_A = A * rdmx
#     print(sampled_A)
    return sampled_A

def computeNcoe(scale,layers):
    num = 0
    for i in range(layers):
        num = num + pow(scale, i)
    return num

# def GSTgenerator_Pyramid(data, variation, args):
#     bar = Bar('>>>', fill='>', max=len(data))
#     label_all = np.zeros(len(data))

#     num_coe = computeNcoe(args.numScales, args.numLayers)
#     GSTcoe_all_p = np.zeros((len(data),1,num_coe,variation))

#     for k,(g, labels) in enumerate(data):
#         label_all[k] = int(labels.item())
#         A = np.zeros((g.num_nodes(),g.num_nodes()))
#         for i in range(g.num_edges()):
#             A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1
#         node_attr = np.expand_dims(np.expand_dims(g.ndata['node_attr'].numpy(), axis=0), axis=0)
#         A_multiscale = A.copy()
#         node_attr_multiscale = node_attr.copy()

#         for ptimes in range(variation):

#             GSTmodel = np_GST.DiffusionScattering(args.numScales, args.numLayers, A_multiscale)        
#             # print('node_attr_multiscale',np.shape(node_attr_multiscale))
#             co_GST = GSTmodel.computeTransform(node_attr_multiscale)

#             num_coe = np.shape(co_GST)[2]

#             GSTcoe_all_p[k,:,:,ptimes] = co_GST[0]
#             # GSTcoe_all[k] = co_GST[0]
            
#             node_sampled, node_sampled_c = downsample(A_multiscale, method='umax', laplacian_basic = 'unnormalized', trick = 0)
#             Laplacian = laplacian(A_multiscale, laplacian_basic = 'unnormalized')
#             R_Laplacian = Kron_Reduction(Laplacian, node_sampled, node_sampled_c)
#             sampled_lenth = np.shape(R_Laplacian)[0]
#             A_multiscale = np.multiply(R_Laplacian,(np.eye(sampled_lenth)-np.ones((sampled_lenth,sampled_lenth))))
#             A_multiscale[np.isnan(A_multiscale)] = 0
#             node_attr_multiscale = node_attr_multiscale[:,:,np.squeeze(node_sampled, axis=0)]
#             node_attr_multiscale[np.isnan(node_attr_multiscale)] = 0

#         bar.next()
#     bar.finish()
#     # np.save('/DATA7_DB7/data/gjliu/dataset/'+'PROTEINS'+'/allphi_p_'+'PROTEINS'+'.npy',GSTcoe_all_p)
#     GSTcoe_all = GSTcoe_all_p.reshape(np.shape(GSTcoe_all_p)[0],np.shape(GSTcoe_all_p)[1],np.shape(GSTcoe_all_p)[2]*np.shape(GSTcoe_all_p)[3])
#     return  GSTcoe_all, label_all #

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

if __name__ == '__main__':
    data_source = 'dataset' # 'map' or 'dataset'
    ## 载入graph
    dataFile = '/DB/rhome/gjliu/workspace/graph_scattering/test_graph/temp.mat'

    if data_source == 'map':
        data = scio.loadmat(dataFile)
        coords = data['coords']
        A = np.power(data['DIST'], -1)
        A[np.isinf(A)] = 0
    elif data_source == 'dataset':
        # data = dgl.data.TUDataset('CORA')
        data = dgl.data.CoraGraphDataset()
        # print(data)
        themax = int(0)
        # for k,(g, labels) in enumerate(data):
        #     if g.num_nodes() > themax:
        #         themax = g.num_nodes()
        #         loc = k
        g = data[0]
        print('Number of num_nodes:', g.num_nodes())
        print('Number of num_edges:', g.num_edges())
        A = np.zeros((g.num_nodes(),g.num_nodes()))
        for i in range(g.num_edges()):
            A[g.edges()[0][i].item()][g.edges()[1][i].item()] = 1
        node_features = g.ndata['feat']
        node_attr = np.expand_dims(np.expand_dims(node_features, axis=0), axis=0)
        print(np.shape(node_features))
        node_features = np.dot(A, node_attr[0,0,:,0])
        node_attr = np.expand_dims(np.expand_dims(node_features, axis=0), axis=0)
        draw_graph(A, node_attr)
        plt.savefig('/DB/rhome/gjliu/workspace/graph_scattering/test_graph/test0.jpg')