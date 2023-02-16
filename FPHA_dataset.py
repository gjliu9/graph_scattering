import os
from os.path import join as pjoin
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import math
import pickle

class FPHADataset(Dataset):
    def __init__(self, dataroot,split,thres=0.002,normalize=True):
        self.dataroot = dataroot
        self.thres = thres
        self.normalize = normalize
        phis = np.load(pjoin(self.dataroot,'allphi_s2.npy'))
        labels = np.load(pjoin(self.dataroot,'seq_action_label.npy'))
        other_info_path = pjoin(self.dataroot, 'other_info.pkl')
        with open(other_info_path,'rb') as f:
            other_info = pickle.load(f)
        _, train_idx, test_idx = other_info['timeNum'],other_info['train_idx'],other_info['test_idx']
        sample_num = len(phis)

        energy = np.load(pjoin(self.dataroot,'energy_v1_train.npy'))
        avgenergy = np.mean(energy,axis=(0,2))
        reserveNode = get_reserve_idx(avgenergy, thres)
        self.nodeNum = len(reserveNode)

        phis = phis[:,reserveNode,:,:].reshape(sample_num,-1).astype(np.float32)
        if self.normalize:
            phis_mean = np.mean(phis[train_idx],axis=0)
            phis_std = np.std(phis[train_idx], axis=0)
            phis = (phis - phis_mean) / phis_std
            phis[np.isnan(phis)] = 0 # phis_std may be zero, remove invalid values here
            phis[np.isinf(phis)] = 0

        if split == 'train':
            self.phis = phis[train_idx]
            self.labels = labels[train_idx]
        elif split == 'test':
            self.phis = phis[test_idx]
            self.labels = labels[test_idx]
        else:
            raise RuntimeError('Invalid split')

    def __getitem__(self, index):
        return self.phis[index], self.labels[index]
    
    def __len__(self):
        return len(self.phis)


def compute_tree_idx(numScales, cur_node_layer, cur_node_layer_order):
    # numScale : J
    # cur_node_layer: 0, 1, 2
    # cur_node_layer_order: 0, 1, 2
    if cur_node_layer == 0:
        return 0
    else:
        node_layers = np.sum(numScales ** np.arange(0, cur_node_layer))
        return node_layers + cur_node_layer_order
def find_parent_children(tree_idx, numScales, numLayers):
    node_layers = np.cumsum(numScales ** np.arange(0, numLayers)) - 1
    cur_node_layer = 0
    for i in range(node_layers.size - 1):
        if node_layers[i] < tree_idx <= node_layers[i + 1]:
            cur_node_layer = i + 1
            break
    if cur_node_layer == 0:
        parent_node = None
        children_node = np.arange(1, numScales+1, dtype=int)
    elif cur_node_layer == numLayers - 1:
        cur_node_layer_order = tree_idx - node_layers[cur_node_layer - 1] - 1
        parent_node_layer = cur_node_layer - 1
        parent_node_layer_order = math.floor(cur_node_layer_order / numScales)
        parent_node = compute_tree_idx(numScales, parent_node_layer, parent_node_layer_order)
        children_node = None
    else:
        cur_node_layer_order = tree_idx - node_layers[cur_node_layer - 1] - 1
        parent_node_layer = cur_node_layer - 1
        parent_node_layer_order = math.floor(cur_node_layer_order / numScales)
        parent_node = compute_tree_idx(numScales, parent_node_layer, parent_node_layer_order)
        children_node_layer = cur_node_layer + 1
        children_node = []
        for i in range(numScales):
            children_node.append(compute_tree_idx(numScales, children_node_layer, cur_node_layer_order*numScales + i))
        children_node = np.array(children_node, dtype=int)
    return parent_node, children_node
def get_reserve_idx(avgenergy,thres):

    nodeNum = len(avgenergy)
    reserveNode = set()
    reserveNode.add(0)
    for idx in range(1,nodeNum):
    #     print('idx',idx)
        parent,_ = find_parent_children(idx, 100, 3)
        if parent != None:
    #         print(parent)
            parent_energy = avgenergy[parent]
            now_energy = avgenergy[idx]
            if parent in reserveNode and now_energy/parent_energy >= thres:
                reserveNode.add(idx)
    reserveNode = np.array(sorted(list(reserveNode)))
    return reserveNode