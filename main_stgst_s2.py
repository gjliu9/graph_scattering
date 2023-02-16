#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2021 chaopan2 <chaopan2@illinois.edu>
#
# Distributed under terms of the MIT license.

# system package
from os import listdir
from os.path import join
import numpy as np
import time
import torch
import math
import argparse
# our package
from Modules.STGST_torch_s2 import STGSTModule
import Modules.graphScattering as np_GST
from sklearn.ensemble import RandomForestClassifier
import pickle
import math
def equal(a,b,thres):
    return np.all(np.abs(a-b) < thres)
def read_FPHA(dataDir,frame_interval):
    points3d_path = join(dataDir, 'seq_points3d.npy')
    label_path = join(dataDir,'seq_action_label.npy')
    other_info_path = join(dataDir, 'other_info.pkl')
    points3d = np.load(points3d_path) # points3d [B, T, 21, 3]
    points3d = points3d[:,::frame_interval,:,:] # [B, (T/frame_interval), 21, 3]
    if args.normalize_cord == 1:
        print('Normalizing')
        for seq_idx in range(len(points3d)):
            seq = points3d[seq_idx] # [(T/frame_interval), 21, 3]
            seq_min = np.amin(seq, axis=(0,1))
            seq_max = np.amax(seq, axis=(0,1))
            assert seq_min.shape == seq_max.shape == (3,)
            seq_mid = (seq_min + seq_max)/2
            new_seq = (seq - seq_mid) / (seq_max-seq_mid)
            points3d[seq_idx] = new_seq
    elif args.normalize_cord == 2: # 相对手腕节点normalize
        print('Normalizing cord, Mode 2')
        points3d = points3d - points3d[:,:,0:1,:]

    labels = np.load(label_path)
    with open(other_info_path,'rb') as f:
        other_info = pickle.load(f)
    timeNum, train_idx, test_idx = other_info['timeNum'],other_info['train_idx'],other_info['test_idx']
    timeNum_interval = math.ceil(timeNum/frame_interval)
    assert timeNum_interval == points3d.shape[1]
    return points3d,labels,timeNum_interval, train_idx, test_idx


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description="ST-GST configuration")
    parser.add_argument("--jointNum", type=int, default=21, help="Number of joints in skeleton graph")
    parser.add_argument("--numScales_space", type=int, default=5, help="Number of scales for spatial layer")
    parser.add_argument("--numScales_time", type=int, default=20, help="Number of scales for temporal layer")
    parser.add_argument("--numLayers", type=int, default=3, help="Number of layers")
    parser.add_argument("--dataDir", type=str, default='/DB/rhome/zdcheng/workspace/STGST/FPHA_process/FPHA_processed_all',
                        help="Data path")
    # parser.add_argument("--dataValidity", type=str, default='skeleton_data_validity.npy',
    #                     help="Check if the samples are valid")
    parser.add_argument("--batchSize", type=int, default=5,
                        help="Batch size of data when computing the ST-GST coefficients")
    parser.add_argument("--frame_interval", type=int)
    parser.add_argument("--n_estimator", type=int)
    parser.add_argument("--normalize_cord", type=int)
    parser.add_argument("--normalize_coeff", type=int)
    args = parser.parse_args()

    # Number of scales in one composite layer
    numScales_tree = args.numScales_space * args.numScales_time
    # Splitting method. This split is used by previous benchmarks
    # skeleton_data_valid = np.load(args.dataValidity)
    # train_sub = [0, 1, 2, 4, 9]
    # test_sub = [3, 5, 6, 7, 8]

    # load data
    # MSR_data_dir = 'dataset/MSRAction3D/MSRAction3DSkeleton(20joints)/'
    # MSR_data_dir = 'dataset/MSRAction3D/MSRAction3DSkeletonReal3D/'
    # data contain the positions of each sample, labels record the action type, and time_num is the number of
    # time steps in each sample
    data, labels, timeNum, train_idx, test_idx = read_FPHA(args.dataDir, args.frame_interval) # data B * T * 21 * 3
    data = np.transpose(data,(0,3,1,2)) # B * 3 * T * 21
    print(data.shape)
    
    # check train and test set do not overlap
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    print('Data size:', data.shape, 'Number of training samples:', len(train_idx),
          'Number of testing samples:', len(test_idx))
    print('=' * 20)
    assert data.shape[0] == len(train_idx) + len(test_idx)

    # Number of tree nodes before pruning
    N_tree_nodes = np.int(np.sum(numScales_tree ** np.arange(0, args.numLayers)))
    sample_num = labels.shape[0]
    coord_dim = 3  # coord dim = 3
    # This is the predefined skeleton graph
    joint_connection = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),
                (11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20))

    # This is the simplest way to build the graph, one can change the graph initialization to achieve better performance
    joint_adj = np.zeros((args.jointNum, args.jointNum)).astype('float32')
    # Just use a simple graph with unit weight
    for i in range(len(joint_connection)):
        joint_adj[joint_connection[i][0], joint_connection[i][1]] = 1.0
        joint_adj[joint_connection[i][1], joint_connection[i][0]] = 1.0

    time_adj = np.zeros((timeNum, timeNum)).astype('float32')
    # acyclic temporal line graph with unit weight
    for i in range(timeNum - 1):
        time_adj[i, i + 1] = 1
        time_adj[i + 1, i] = 1

    # ST-GST computation
    start_time = time.time()
    model = STGSTModule(args.numScales_space,
                        args.numScales_time,
                        args.numLayers,
                        joint_adj,
                        time_adj).cuda()
    # xGST stores all computed coefficients. The size may change for different pooling method used in model. If we
    # are averaging over time domain, the size of output for each sample should be
    # coord_dim * N_tree_nodes * args.jointNum
    xSTGST = np.empty([sample_num, N_tree_nodes, coord_dim, args.jointNum])
    energy = np.empty([sample_num, N_tree_nodes, coord_dim]) #[B, 10101, 3]
    # Larger batchSize uses more CUDA memory, but faster. Current version of code always assumes that GPU is available
    # for computation. If not, please contact me and I can change it back to numpy version
    for batch_iter in range(math.ceil(sample_num / args.batchSize)):
        start = time.time()
        start_idx = batch_iter * args.batchSize
        end_idx = min((batch_iter + 1) * args.batchSize, sample_num)
        sample = data[start_idx: end_idx, :, :, :]
        torch_sample = torch.from_numpy(sample.astype('float32')).cuda()
        # The shape of input torch_sample should be [batchSize, channel, time_num, joint_num]. channel is normally 3 if
        # 3D coordinates are provided

        batch_phi,batch_rho = model.forward(torch_sample, N_tree_nodes) #batch_rho [B,10101,3,T,N]
        xSTGST[start_idx: end_idx, :, :] = batch_phi.detach().cpu().numpy()
        batch_energy = torch.norm(batch_rho, dim=(3,4)).detach().cpu().numpy() # [B,10101,3]
    
        energy[start_idx:end_idx,:] = batch_energy
        print('Processing batch', batch_iter)
    print('Time used for computing coefficients:', time.time() - start_time)
    print('=' * 20)

    # Performance only using separable filters
    np.save('allphi_s2.npy',xSTGST) # [B, 10101, 3, N]
    energy_train = energy[train_idx,:,:]
    np.save('energy_v1_train.npy',energy_train)

    xSTGST = xSTGST.reshape((sample_num, -1))
    if args.normalize_coeff == 1:
        print('Normalizing phis')
        xTrain = xSTGST[train_idx, :]
        x_mean = np.mean(xTrain, axis=0)
        x_std = np.std(xTrain, axis=0)
        xSTGST = (xSTGST - x_mean) / x_std
        xSTGST[np.isnan(xSTGST)] = 0 # x_std may be zero, remove invalid values here
        xSTGST[np.isinf(xSTGST)] = 0

    # Classification using random forest classifier
    # Training
    xTrain = xSTGST[train_idx, :]
    yTrain = labels[train_idx]
    classifier = RandomForestClassifier(n_estimators=300)
    print('RF training starts')
    start_time = time.time()
    classifier.fit(xTrain, yTrain)
    # Testing
    xTest = xSTGST[test_idx, :]
    yTest = labels[test_idx]
    print('Classification accuracy:', classifier.score(xTest, yTest))
    print('Time used for RF classification:', time.time() - start_time)
    print('=' * 20)
    ################################################
    # Kronecker product joint scattering
    # kron_start = time.time()
    # numMoments = 4
    # numScales_kron = 15
    # numLayers_kron = 3
    # nFeatures_kron = np.int(np.sum(numScales_kron ** np.arange(0, numLayers_kron)) * numMoments)
    # xGST_kron = np.empty([sample_num, coord_dim, nFeatures_kron])
    # model_kron = np_GST.GeometricScattering(numScales_kron, numLayers_kron, numMoments, np.kron(joint_adj, time_adj))
    # batchSize = 100
    # for batch_iter in range(math.ceil(sample_num / batchSize)):
    #     start = time.time()
    #     start_idx = batch_iter * batchSize
    #     end_idx = min((batch_iter + 1) * batchSize, sample_num)
    #     sample = data[start_idx: end_idx, :, :, :].transpose((0, 1, 3, 2))
    #     sample = sample.reshape((end_idx - start_idx, coord_dim, -1))
    #     xGST_kron[start_idx: end_idx, :, :] = model_kron.computeTransform(sample)
    #     print('Processing batch', batch_iter)
    # print('Time used for computing coefficients on Kronecker product graph:', time.time() - kron_start)
    # print('=' * 20)

    # Performance using both separable and joint filters
    # xGST = np.concatenate((xSTGST, xGST_kron), axis=2)
    # xGST = xGST.reshape((sample_num, -1))
    # # Training
    # xTrain = xGST[train_idx, :]
    # yTrain = labels[train_idx]
    classifier = RandomForestClassifier(n_estimators=300)
    # print('RF training starts')
    # start_time = time.time()
    # classifier.fit(xTrain, yTrain)
    # # Testing
    # xTest = xGST[test_idx, :]
    # yTest = labels[test_idx]
    # print('Classification accuracy using two designs:', classifier.score(xTest, yTest), time.time() - start_time)


