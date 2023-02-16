#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2021 chaopan2 <chaopan2@illinois.edu>
#
# Distributed under terms of the MIT license.

import numpy as np
import torch
import math
import Utils.graphTools as graphTools
import torch.nn as nn

zeroTolerance = 1e-7  # Values below this number are considered zero.
infiniteNumber = 1e12  # infinity equals this number

def diffusionWavelets(J, T):
    """
    diffusionWavelets: computes the filter bank for the diffusion wavelets

    See R. R. Coifman and M. Maggioni, “Diffusion wavelets,” Appl. Comput.
        Harmonic Anal., vol. 21, no. 1, pp. 53–94, July 2006.
    Alternatively, see eq. (6) of F. Gama, A. Ribeiro, and J. Bruna, “Diffusion
        Scattering Transforms on Graphs,” in 7th Int. Conf. Learning
        Representations. New Orleans, LA: Assoc. Comput. Linguistics,
        6-9 May 2019, pp. 1–12.

    Input:
        J (int): number of scales
        T (np.array): lazy diffusion operator

    Output:
        H (np.array): of shape J x N x N contains the J matrices corresponding
            to all the filter scales
    """
    # J is the number of scales, and we do wavelets from 0 to J-1, so it always
    # needs to be at least 1
    assert J > 0
    N = T.shape[0]  # Number of nodes
    assert T.shape[1] == N  # Check it's a square matrix
    I = np.eye(N)  # Identity matrix
    H = (I - T).reshape(1, N, N)  # 1 x N x N
    for j in range(1, J):
        thisPower = 2 ** (j-1)  # 2^(j-1)
        powerT = np.linalg.matrix_power(T, thisPower)  # T^{2^{j-1}}
        thisH = powerT @ (I - powerT)  # T^{2^{j-1}} * (I - T^{2^{j-1}})
        H = np.concatenate((H, thisH.reshape(1, N, N)), axis=0)
    return H

def diffusionWavelets_torch(J, T):
    assert J > 0
    N = T.shape[0]
    assert T.shape[1] == N
    tmpT = torch.from_numpy(T.astype(np.float32)).cuda()
    I = torch.eye(N).cuda()
    H = (I-tmpT).reshape(1,N,N)
    for j in range(1,J):
        thisPower = 2**(j-1)
        powerT = torch.matrix_power(tmpT, thisPower)
        thisH = torch.matmul(powerT, I-powerT )
        H = torch.cat((H,thisH.reshape(1,N,N)), dim=0)
    return H

def AdjToP(adj):
    d = np.sum(adj, axis=1)
    killIndices = np.nonzero(d < zeroTolerance)[0]
    dReady = d.copy()
    dReady[killIndices] = 1.
    dInv = 1. / dReady
    dInv[killIndices] = 0.
    Dinv = np.diag(dInv)
    P = 1 / 2 * (np.eye(adj.shape[0]) + adj.dot(Dinv))
    P = P.astype(np.float32)
    return P

class STGSTModule(nn.Module):
    def __init__(self,numScales_space, numScales_time, numLayers, w_space, w_time):
        super(STGSTModule,self).__init__()

        self.J_space = numScales_space
        self.J_time = numScales_time
        self.J = numScales_space * numScales_time
        self.L = numLayers
        self.W_space = w_space.copy()
        self.W_time = w_time.copy()
        self.N_space = w_space.shape[0]
        self.N_time = w_time.shape[0]

        self.a_space = nn.Parameter(torch.ones(self.J_space,1,1))
        self.b_space = nn.Parameter(torch.ones(self.J_space,1,1))

        self.a_time = nn.Parameter(torch.ones(self.J_time,1,1))
        self.b_time = nn.Parameter(torch.ones(self.J_time,1,1))

        # space
        self.P_space = torch.tensor(AdjToP(self.W_space)).cuda() # N*N
        self.P_powers1_space = self.P_powers1(self.P_space, self.J_space)
        self.P_powers2_space = self.P_powers2(self.P_space, self.J_space) # J_space * N * N

        # time
        self.P_time = torch.tensor(AdjToP(self.W_time)).cuda()
        self.P_powers1_time = self.P_powers1(self.P_time, self.J_time)
        self.P_powers2_time = self.P_powers2(self.P_time, self.J_time) # J_time * T * T


        # construct spatial wavelet by Geometric scattering
        d = np.sum(self.W_space, axis=1)
        killIndices = np.nonzero(d < zeroTolerance)[0]
        dReady = d.copy()
        dReady[killIndices] = 1.
        dInv = 1. / dReady
        dInv[killIndices] = 0.
        Dinv = np.diag(dInv)
        self.P = 1 / 2 * (np.eye(self.N_space) + self.W_space.dot(Dinv))
        self.H_space = diffusionWavelets(self.J_space, self.P).astype('float32')
        self.H_space = torch.from_numpy(self.H_space).cuda()  # J_space x N_space x N_space

        # construct temporal wavelet by Geometric scattering
        d = np.sum(self.W_time, axis=1)
        killIndices = np.nonzero(d < zeroTolerance)[0]
        dReady = d.copy()
        dReady[killIndices] = 1.
        dInv = 1. / dReady
        dInv[killIndices] = 0.
        Dinv = np.diag(dInv)
        self.P = 1 / 2 * (np.eye(self.N_time) + self.W_time.dot(Dinv))
        self.H_time = diffusionWavelets(self.J_time, self.P).astype('float32')
        self.H_time = torch.from_numpy(self.H_time).cuda()  # J_time x N_time x N_time

        
    def tmpcheck(self):
        tmp_H_space = self.a_space*self.P_powers1_space - self.b_space*self.P_powers2_space
        tmp_H_time = self.a_time*self.P_powers1_time - self.b_time*self.P_powers2_time
        np.save('H_space_new.npy',tmp_H_space.detach().cpu().numpy())
        np.save('H_time_new.npy',tmp_H_time.detach().cpu().numpy())
        raise RuntimeError()


    def P_powers1(self, P_ori, J):
        N = P_ori.shape[0]
        P_powers = torch.eye(N).reshape(1, N, N).cuda()
        for j in range(1,J):
            tmp_P_power = torch.matrix_power(P_ori,2**(j-1)).reshape(1,N,N)
            P_powers = torch.cat((P_powers,tmp_P_power), dim=0)
        return P_powers

    def P_powers2(self, P_ori, J):
        N = P_ori.shape[0]
        P_powers = P_ori.reshape(1,N,N)
        for j in range(1,J):
            tmp_P_power = torch.matrix_power(P_ori,2**j).reshape(1,N,N)
            P_powers = torch.cat((P_powers,tmp_P_power), dim=0)
        return P_powers

    def forward(self, torch_x, N_tree_nodes):
        # Averaging over temporal domain. Default transform method
        # dimension of torch_x: batchSize x F(3) x N_time x N_space
        H_space = self.a_space*self.P_powers1_space - self.b_space*self.P_powers2_space
        # print(H_space.shape)
        H_time = self.a_time*self.P_powers1_time - self.b_time*self.P_powers2_time
        # print(H_time.shape)

        assert len(torch_x.shape) == 4
        B = torch_x.shape[0]
        F = torch_x.shape[1]
        Phi = torch.empty([B, N_tree_nodes, F, self.N_space]).cuda() # [B, 10101, 3, N]
        Phi[:, 0, :, :] = torch.mean(torch_x, 2) # torch_x [B,3,N]
        Phi_count = 1
        nextRhoHx = torch.empty([B, N_tree_nodes, F, self.N_time, self.N_space]).cuda()  # [B, 10101, 3, T, N]
        nextRhoHx[:, 0, :, :, :] = torch_x
        nextRhoHx_count = 1
        start_idx = 0

        for l in range(1, self.L):  # l = 1,2,...,L-1
            for j in range(self.J ** (l - 1)):  # j = 0,...,J**(l-1)-1
                # B x F x N_time x N_space, at cuda
                # this is the signal at their parent node
                thisX = nextRhoHx[:, start_idx + j, :, :, :]  # [B, 3, T, N]
                for space_scale_itr in range(self.J_space):
                    # thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
                    thisX_after_spatial_gst = torch.matmul(thisX, self.H_space[space_scale_itr, :, :])  # [B, 3, T, N]

                    gstX_allJoint = thisX_after_spatial_gst.permute(0,3,1,2).unsqueeze(2) # [B, N, 1, 3, T]
                    RhoHX_allJoint = torch.abs(torch.matmul(gstX_allJoint, self.H_time))  # [B, N, J_time, 3, T]
                    # RhoHX_allJoint = RhoHX_allJoint.reshape(B, self.N_space, -1, F, self.N_time)  # [B*N, J_time, 3, T] -> # [B, N, J_time, 3, T]
                    RhoHX_allJoint = RhoHX_allJoint.permute(0,2,3,4,1) # [B, N, J_time, 3, T] -> [B, J_time, 3, T, N]

                    nextRhoHx[:, nextRhoHx_count: nextRhoHx_count+self.J_time, :, :, :] = RhoHX_allJoint  # [B, 10101, 3, T, N]

                    phi_allnode = torch.mean(RhoHX_allJoint, 3) # [B, J_time, 3, N]
                    Phi[:, Phi_count: Phi_count+self.J_time, :, :] = phi_allnode
                    
                    Phi_count += self.J_time
                    nextRhoHx_count += self.J_time
                
            start_idx += self.J ** (l - 1)

        return Phi,nextRhoHx





class SpatialTemporalScatteringTransformVertexTime:

    def __init__(self, numScales_space, numScales_time, numLayers, w_space, w_time):

        self.J_space = numScales_space
        self.J_time = numScales_time
        self.J = numScales_space * numScales_time
        self.L = numLayers
        self.W_space = w_space.copy()
        self.W_time = w_time.copy()
        self.N_space = w_space.shape[0]
        self.N_time = w_time.shape[0]

        # construct spatial wavelet by Geometric scattering
        d = np.sum(self.W_space, axis=1)
        killIndices = np.nonzero(d < zeroTolerance)[0]
        dReady = d.copy()
        dReady[killIndices] = 1.
        dInv = 1. / dReady
        dInv[killIndices] = 0.
        Dinv = np.diag(dInv)
        self.P = 1 / 2 * (np.eye(self.N_space) + self.W_space.dot(Dinv))
        self.H_space = diffusionWavelets(self.J_space, self.P).astype('float32')
        self.H_space = torch.from_numpy(self.H_space).cuda()  # J_space x N_space x N_space

        # construct temporal wavelet by Geometric scattering
        d = np.sum(self.W_time, axis=1)
        killIndices = np.nonzero(d < zeroTolerance)[0]
        dReady = d.copy()
        dReady[killIndices] = 1.
        dInv = 1. / dReady
        dInv[killIndices] = 0.
        Dinv = np.diag(dInv)
        self.P = 1 / 2 * (np.eye(self.N_time) + self.W_time.dot(Dinv))
        self.H_time = diffusionWavelets(self.J_time, self.P).astype('float32')
        self.H_time = torch.from_numpy(self.H_time).cuda()  # J_time x N_time x N_time

    def computeTransformSpaceNoAvg(self, torch_x, N_tree_nodes):
        # Averaging over temporal domain. Default transform method
        # dimension of torch_x: batchSize x F(3) x N_time x N_space
        assert len(torch_x.shape) == 4
        B = torch_x.shape[0]
        F = torch_x.shape[1]
        Phi = torch.empty([B, F, N_tree_nodes * self.N_space]).cuda()
        Phi[:, :, 0: self.N_space] = torch.mean(torch_x, 2)  # batchSize x F(3) x N_space
        Phi_count = self.N_space
        nextRhoHx = torch.empty([B, N_tree_nodes, F, self.N_time, self.N_space]).cuda()  # at cuda
        nextRhoHx[:, 0, :, :, :] = torch_x
        nextRhoHx_count = 1
        start_idx = 0

        for l in range(1, self.L):  # l = 1,2,...,L
            for j in range(self.J ** (l - 1)):  # j = 0,...,l-1
                # B x F x N_time x N_space, at cuda
                # this is the signal at their parent node
                thisX = nextRhoHx[:, start_idx + j, :, :, :]  # B x F x N_time x N_space
                for space_scale_itr in range(self.J_space):
                    thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
                    for time_itr in range(self.N_time):
                        thisX_thisTime = thisX[:, :, time_itr, :]  # B x F x N_space
                        thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
                                                                                  self.H_space[space_scale_itr, :, :])
                    # sub-layer after spatial gst
                    for space_itr in range(self.N_space):
                        gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr].reshape((B, 1, F, self.N_time))
                        RhoHX_thisJoint = torch.abs(torch.matmul(gstX_thisJoint, self.H_time))
                        nextRhoHx[:, nextRhoHx_count: nextRhoHx_count+self.J_time, :, :, space_itr] = RhoHX_thisJoint

                        # compute the aggregation
                        phi_node = torch.mean(RhoHX_thisJoint, 3).transpose(1, 2)  # B x F x 2
                        Phi[:, :, Phi_count: Phi_count+self.J_time] = phi_node
                        Phi_count += self.J_time

                    nextRhoHx_count += self.J_time

            start_idx += self.J ** (l - 1)

        return Phi

# class GeometricScattering:
#     """
#     GeometricScattering: geometric scattering transform

#     Initialization:

#     Input:
#         numScales (int): number of wavelet scales (size of the filter bank)
#         numLayers (int): number of layers
#         numMoments (int): number of moments to compute invariants
#         adjacencyMatrix (np.array): of shape N x N

#     Output:
#         Instantiates the class for the geometric scattering transform

#     Methods:

#         Phi = .computeTransform(x): computes the diffusion scattering
#             coefficients of input x (np.array of shape B x F x N, with B the
#             batch size, F the number of node features, and N the number of
#             nodes)
#     """

#     def __init__(self, numScales, numLayers, numMoments, adjacencyMatrix):

#         self.J = numScales
#         self.L = numLayers
#         assert numMoments > 0
#         self.Q = numMoments
#         self.W = adjacencyMatrix.copy()
#         self.N = self.W.shape[0]
#         assert self.W.shape[1] == self.N

#         d = np.sum(self.W, axis=1)
#         # Nodes with zero degree or negative degree (there shouldn't be any since
#         # (i) the graph is connected and (ii) all edge weights are supposed to be positive)
#         killIndices = np.nonzero(d < zeroTolerance)[0]
#         dReady = d.copy()
#         dReady[killIndices] = 1.
#         dInv = 1./dReady
#         # Put back zeros in those numbers that had been failing
#         dInv[killIndices] = 0.
#         Dinv = np.diag(dInv)
#         # Lazy diffusion random walk
#         self.P = 1/2*(np.eye(self.N) + self.W.dot(Dinv))
#         # Construct wavelets
#         self.H = diffusionWavelets(self.J, self.P)
#         #   Note that the diffusion wavelets coefficients don't change. What
#         #   changes is the matrix (now it's the lazy diffusion random walk
#         #   instead of the lazy diffusion adjacency), but nothing else.

#     def computeMoments(self, x):

#         # The input is B x J x F x N and the output has to be B x J x F x Q
#         # (J is the number of scales we have up to here, it doesn't matter)
#         assert len(x.shape) == 4
#         assert x.shape[3] == self.N

#         # Because we have checked that Q is at least 1, we know that the first
#         # order moment we will always be here, so we just compute it
#         Sx = np.sum(x, axis=3)  # B x J x F
#         # Add the dim, because on that dim we will concatenate the values of Q
#         Sx = np.expand_dims(Sx, 3)  # B x J x F x 1
#         # Now, for all other values of Q that we haven't used yet
#         for q in range(2, self.Q+1):  # q = 2, ..., Q
#             # Compute the qth moment and add up
#             thisMoment = np.sum(x ** q, axis=3)  # B x J x F
#             # Add the extra dimension
#             thisMoment = np.expand_dims(thisMoment, 3)  # B x J x F x 1
#             # Concatenate to the already existing Sx
#             Sx = np.concatenate((Sx, thisMoment), axis=3)  # B x J x F x q

#         return Sx  # B x J x F x Q

#     def computeTransform(self, x):
#         # Check the dimension of x: batchSize x numberFeatures x numberNodes
#         assert len(x.shape) == 3
#         B = x.shape[0]  # batchSize
#         F = x.shape[1]  # numberFeatures
#         assert x.shape[2] == self.N
#         # Start creating the output
#         # Compute the first coefficients
#         Phi = self.computeMoments(np.expand_dims(x, 1))  # B x 1 x F x Q
#         Phi = Phi.squeeze(1)  # B x F x Q
#         # Reshape x to account for the increasing J dimension that we will have
#         rhoHx = x.reshape(B, 1, F, self.N)  # B x 1 x F x N
#         # Now, we move to the rest of the layers
#         for l in range(1, self.L):  # l = 1,2,...,L
#             nextRhoHx = np.empty([B, 0, F, self.N])
#             for j in range(self.J ** (l-1)):  # j = 0,...,l-1
#                 thisX = rhoHx[:, j, :, :]  # B x 1 x F x N
#                 thisHx = thisX.reshape(B, 1, F, self.N) @ self.H.reshape(1, self.J, self.N, self.N)  # B x J x F x N
#                 thisRhoHx = np.abs(thisHx)  # B x J x F x N
#                 nextRhoHx = np.concatenate((nextRhoHx, thisRhoHx), axis=1)

#                 phi_j = self.computeMoments(thisRhoHx)  # B x J x F x Q
#                 phi_j = phi_j.transpose(0, 2, 1, 3)  # B x F x J x Q
#                 phi_j = phi_j.reshape(B, F, self.J * self.Q)
#                 Phi = np.concatenate((Phi, phi_j), axis=2)  # Keeps adding the coefficients
#             rhoHx = nextRhoHx.copy()

#         return Phi

# def compute_tree_idx(numScales, cur_node_layer, cur_node_layer_order):
#     if cur_node_layer == 0:
#         return 0
#     else:
#         node_layers = np.sum(numScales ** np.arange(0, cur_node_layer))
#         return node_layers + cur_node_layer_order

# def find_parent_children(tree_idx, numScales, numLayers):
#     node_layers = np.cumsum(numScales ** np.arange(0, numLayers)) - 1
#     cur_node_layer = 0
#     for i in range(node_layers.size - 1):
#         if node_layers[i] < tree_idx <= node_layers[i + 1]:
#             cur_node_layer = i + 1
#             break
#     if cur_node_layer == 0:
#         parent_node = None
#         children_node = np.arange(1, numScales+1, dtype=int)
#     elif cur_node_layer == numLayers - 1:
#         cur_node_layer_order = tree_idx - node_layers[cur_node_layer - 1] - 1
#         parent_node_layer = cur_node_layer - 1
#         parent_node_layer_order = math.floor(cur_node_layer_order / numScales)
#         parent_node = compute_tree_idx(numScales, parent_node_layer, parent_node_layer_order)
#         children_node = None
#     else:
#         cur_node_layer_order = tree_idx - node_layers[cur_node_layer - 1] - 1
#         parent_node_layer = cur_node_layer - 1
#         parent_node_layer_order = math.floor(cur_node_layer_order / numScales)
#         parent_node = compute_tree_idx(numScales, parent_node_layer, parent_node_layer_order)
#         children_node_layer = cur_node_layer + 1
#         children_node = []
#         for i in range(numScales):
#             children_node.append(compute_tree_idx(numScales, children_node_layer, cur_node_layer_order*numScales + i))
#         children_node = np.array(children_node, dtype=int)
#     return parent_node, children_node

    # def computeTransformNoAvg(self, torch_x, N_tree_nodes):
    #     # No pooling is performed. Cost large amount of memory
    #     # dimension of torch_x: batchSize x F(3) x N_time x N_space
    #     assert len(torch_x.shape) == 4
    #     B = torch_x.shape[0]
    #     F = torch_x.shape[1]
    #     nextRhoHx = torch.empty([B, N_tree_nodes, F, self.N_time, self.N_space]).cuda()  # at cuda
    #     nextRhoHx[:, 0, :, :, :] = torch_x
    #     nextRhoHx_count = 1
    #     start_idx = 0

    #     for l in range(1, self.L):  # l = 1,2,...,L
    #         for j in range(self.J ** (l - 1)):  # j = 0,...,l-1
    #             # B x F x N_time x N_space, at cuda
    #             # this is the signal at their parent node
    #             thisX = nextRhoHx[:, start_idx + j, :, :, :]  # B x F x N_time x N_space
    #             for space_scale_itr in range(self.J_space):
    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = thisX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr].reshape((B, 1, F, self.N_time))
    #                     RhoHX_thisJoint = torch.abs(torch.matmul(gstX_thisJoint, self.H_time))
    #                     nextRhoHx[:, nextRhoHx_count: nextRhoHx_count+self.J_time, :, :, space_itr] = RhoHX_thisJoint

    #                 nextRhoHx_count += self.J_time

    #         start_idx += self.J ** (l - 1)

    #     return nextRhoHx.transpose(1, 2).reshape(B, F, -1).cpu().numpy()



    # def computeTransform(self, torch_x, N_tree_nodes):
    #     # Both space and time are averaged. Lose to much information in this case
    #     # dimension of torch_x: batchSize x F(3) x N_time x N_space
    #     assert len(torch_x.shape) == 4
    #     B = torch_x.shape[0]
    #     F = torch_x.shape[1]
    #     nextRhoHx = torch.empty([B, N_tree_nodes, F, self.N_time, self.N_space]).cuda()  # at cuda
    #     nextRhoHx[:, 0, :, :, :] = torch_x
    #     nextRhoHx_count = 1
    #     start_idx = 0

    #     for l in range(1, self.L):  # l = 1,2,...,L
    #         for j in range(self.J ** (l - 1)):  # j = 0,...,l-1
    #             # B x F x N_time x N_space, at cuda
    #             # this is the signal at their parent node
    #             thisX = nextRhoHx[:, start_idx + j, :, :, :]  # B x F x N_time x N_space
    #             for space_scale_itr in range(self.J_space):
    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = thisX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr].reshape((B, 1, F, self.N_time))
    #                     RhoHX_thisJoint = torch.abs(torch.matmul(gstX_thisJoint, self.H_time))
    #                     nextRhoHx[:, nextRhoHx_count: nextRhoHx_count+self.J_time, :, :, space_itr] = RhoHX_thisJoint
    #                 nextRhoHx_count += self.J_time
    #         start_idx += self.J ** (l - 1)

    #     # compute the aggregation Phi
    #     Phi = torch.empty([B, F, N_tree_nodes]).cuda()
    #     for i in range(N_tree_nodes):
    #         Phi[:, :, i] = torch.mean(nextRhoHx[:, i, :, :, :].reshape(B, F, -1), 2)
    #     return Phi.cpu().numpy()

    # def computeTransformReluSpaceNoAvg(self, torch_x, N_tree_nodes):
    #     # Change absolute value activation to relu. Does not work well
    #     # dimension of torch_x: batchSize x F(3) x N_time x N_space
    #     assert len(torch_x.shape) == 4
    #     B = torch_x.shape[0]
    #     F = torch_x.shape[1]
    #     Phi = torch.empty([B, F, N_tree_nodes * self.N_space]).cuda()
    #     Phi[:, :, 0: self.N_space] = torch.mean(torch_x, 2)  # batchSize x F(3) x N_space
    #     Phi_count = self.N_space
    #     nextRhoHx = torch.empty([B, N_tree_nodes, F, self.N_time, self.N_space]).cuda()  # at cuda
    #     nextRhoHx[:, 0, :, :, :] = torch_x
    #     nextRhoHx_count = 1
    #     start_idx = 0
    #     m = torch.nn.ReLU()
    #     for l in range(1, self.L):  # l = 1,2,...,L
    #         for j in range(self.J ** (l - 1)):  # j = 0,...,l-1
    #             # B x F x N_time x N_space, at cuda
    #             # this is the signal at their parent node
    #             thisX = nextRhoHx[:, start_idx + j, :, :, :]  # B x F x N_time x N_space
    #             for space_scale_itr in range(self.J_space):
    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = thisX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr].reshape((B, 1, F, self.N_time))
    #                     RhoHX_thisJoint = m(torch.matmul(gstX_thisJoint, self.H_time))
    #                     nextRhoHx[:, nextRhoHx_count: nextRhoHx_count+self.J_time, :, :, space_itr] = RhoHX_thisJoint

    #                     # compute the aggregation
    #                     phi_node = torch.mean(RhoHX_thisJoint, 3).transpose(1, 2)  # B x F x 2
    #                     Phi[:, :, Phi_count: Phi_count+self.J_time] = phi_node
    #                     Phi_count += self.J_time

    #                 nextRhoHx_count += self.J_time

    #         start_idx += self.J ** (l - 1)

    #     return Phi.cpu().numpy()

    # def computeTransformRelu(self, torch_x, N_tree_nodes):
    #     # Both space and time are averaged. Replace abs with relu
    #     # dimension of torch_x: batchSize x F(3) x N_time x N_space
    #     assert len(torch_x.shape) == 4
    #     B = torch_x.shape[0]
    #     F = torch_x.shape[1]
    #     nextRhoHx = torch.empty([B, N_tree_nodes, F, self.N_time, self.N_space]).cuda()  # at cuda
    #     nextRhoHx[:, 0, :, :, :] = torch_x
    #     nextRhoHx_count = 1
    #     start_idx = 0
    #     m = torch.nn.ReLU()
    #     for l in range(1, self.L):  # l = 1,2,...,L
    #         for j in range(self.J ** (l - 1)):  # j = 0,...,l-1
    #             # B x F x N_time x N_space, at cuda
    #             # this is the signal at their parent node
    #             thisX = nextRhoHx[:, start_idx + j, :, :, :]  # B x F x N_time x N_space
    #             for space_scale_itr in range(self.J_space):
    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = thisX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr].reshape((B, 1, F, self.N_time))
    #                     RhoHX_thisJoint = m(torch.matmul(gstX_thisJoint, self.H_time))
    #                     nextRhoHx[:, nextRhoHx_count: nextRhoHx_count+self.J_time, :, :, space_itr] = RhoHX_thisJoint
    #                 nextRhoHx_count += self.J_time
    #         start_idx += self.J ** (l - 1)

    #     # compute the aggregation Phi
    #     Phi = torch.empty([B, F, N_tree_nodes]).cuda()
    #     for i in range(N_tree_nodes):
    #         Phi[:, :, i] = torch.mean(nextRhoHx[:, i, :, :, :].reshape(B, F, -1), 2)
    #     return Phi.cpu().numpy()

    # def computeTransformDoubleReluSpaceNoAvg(self, torch_x, N_tree_nodes):
    #     # Ideas from Professor Ortega. Using splitted positive and negative information to do classification
    #     # dimension of torch_x: batchSize x F(3) x N_time x N_space
    #     assert len(torch_x.shape) == 4
    #     B = torch_x.shape[0]
    #     F = torch_x.shape[1]
    #     m = torch.nn.ReLU()
    #     Phi = torch.empty([B, F, N_tree_nodes * self.N_space * 2]).cuda()
    #     Phi[:, :, 0: self.N_space] = torch.mean(m(torch_x), 2)  # batchSize x F(3) x N_space
    #     Phi[:, :, self.N_space: 2 * self.N_space] = torch.mean(m(-torch_x), 2)
    #     Phi_count = 2 * self.N_space
    #     nextRhoHx = torch.empty([B, N_tree_nodes, F, self.N_time, self.N_space]).cuda()  # at cuda
    #     nextRhoHx[:, 0, :, :, :] = torch_x
    #     nextRhoHx_count = 1
    #     start_idx = 0

    #     for l in range(1, self.L):  # l = 1,2,...,L
    #         for j in range(self.J ** (l - 1)):  # j = 0,...,l-1
    #             # B x F x N_time x N_space, at cuda
    #             # this is the signal at their parent node
    #             thisX = nextRhoHx[:, start_idx + j, :, :, :]  # B x F x N_time x N_space
    #             for space_scale_itr in range(self.J_space):
    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = thisX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr].reshape((B, 1, F, self.N_time))
    #                     RhoHX_thisJoint = torch.matmul(gstX_thisJoint, self.H_time)

    #                     # compute the aggregation
    #                     phi_node = torch.mean(m(RhoHX_thisJoint), 3).transpose(1, 2)
    #                     Phi[:, :, Phi_count: Phi_count + self.J_time] = phi_node
    #                     Phi_count += self.J_time
    #                     phi_node = torch.mean(m(-RhoHX_thisJoint), 3).transpose(1, 2)
    #                     Phi[:, :, Phi_count: Phi_count + self.J_time] = phi_node
    #                     Phi_count += self.J_time

    #                     nextRhoHx[:, nextRhoHx_count: nextRhoHx_count + self.J_time, :, :, space_itr] = \
    #                         torch.abs(RhoHX_thisJoint)

    #                 nextRhoHx_count += self.J_time

    #         start_idx += self.J ** (l - 1)

    #     return Phi.cpu().numpy()

    # def computePrunedTransformLeavesEnergyNoCat(self, torch_x, target_layer, parents_flags):
    #     # Compute the leaves energy based on already known parent flags
    #     # dimension of torch_x: batchSize x F(3) x N_time x N_space
    #     # This part is reserved for Pruned version of this work. Can ignore this part for now.
    #     # See Ioannidis, Vassilis N., Siheng Chen, and Georgios B. Giannakis.
    #     # "Pruned graph scattering transforms." International Conference on Learning Representations. 2019.
    #     assert len(torch_x.shape) == 4
    #     B = torch_x.shape[0]
    #     F = torch_x.shape[1]
    #     # compute how many nodes will be preserved
    #     valid_node_count = sum(parents_flags)
    #     l = target_layer - 1
    #     for j in range(self.J ** l):
    #         cur_node_total_tree_index = compute_tree_idx(self.J, l, j)
    #         if parents_flags[cur_node_total_tree_index] == 1:
    #             valid_node_count += self.J

    #     nextRhoHx = torch.empty([B, valid_node_count, F, self.N_time, self.N_space]).cuda()  # at cuda
    #     nextRhoHx[:, 0, :, :, :] = torch_x
    #     nextRhoHx_count = 1
    #     total_tree_flags = [1]

    #     # layer 1 to target_layer - 1
    #     for l in range(1, target_layer):
    #         for j in range(self.J ** l):
    #             # B x F x N_time x N_space, at cuda
    #             # this is the signal at their parent node
    #             cur_node_total_tree_index = compute_tree_idx(self.J, l, j)
    #             parent_total_tree_index, _ = find_parent_children(cur_node_total_tree_index, self.J, target_layer+1)
    #             if parents_flags[parent_total_tree_index] == 1 and parents_flags[cur_node_total_tree_index] == 1:
    #                 nextRhoHx_parent_index = int(sum(parents_flags[0: parent_total_tree_index]))
    #                 parentX = nextRhoHx[:, nextRhoHx_parent_index, :, :, :]  # B x F x N_time x N_space
    #                 thisRhoHX = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 space_scale_itr = math.floor((j % self.J) / self.J_time)
    #                 time_scale_itr = (j % self.J) % self.J_time

    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = parentX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr]  # B x F x N_time
    #                     RhoHX_thisJoint = torch.abs(torch.matmul(gstX_thisJoint, self.H_time[time_scale_itr, :, :]))
    #                     thisRhoHX[:, :, :, space_itr] = RhoHX_thisJoint

    #                 # nextRhoHx = torch.cat((nextRhoHx, thisRhoHX), dim=1)
    #                 nextRhoHx[:, nextRhoHx_count, :, :, :] = thisRhoHX
    #                 nextRhoHx_count += 1
    #                 total_tree_flags.append(1)
    #             else:
    #                 total_tree_flags.append(0)

    #     # target_layer
    #     l = target_layer
    #     for j in range(0, self.J ** l, self.J):
    #         # B x F x N_time x N_space, at cuda
    #         # this is the signal at their parent node
    #         cur_node_total_tree_index = compute_tree_idx(self.J, l, j)
    #         parent_total_tree_index, _ = find_parent_children(cur_node_total_tree_index, self.J, target_layer+1)
    #         if parents_flags[parent_total_tree_index] == 1:
    #             nextRhoHx_parent_index = int(sum(parents_flags[0: parent_total_tree_index]))
    #             parentX = nextRhoHx[:, nextRhoHx_parent_index, :, :, :]  # B x F x N_time x N_space
    #             thisRhoHX = torch.empty([B, self.J, F, self.N_time, self.N_space]).cuda()
    #             for space_scale_itr in range(self.J_space):
    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = parentX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr].reshape((B, 1, F, self.N_time))
    #                     RhoHX_thisJoint = torch.abs(torch.matmul(gstX_thisJoint, self.H_time))
    #                     thisRhoHX[:, space_scale_itr*self.J_time: (space_scale_itr+1)*self.J_time,
    #                               :, :, space_itr] = RhoHX_thisJoint

    #             # nextRhoHx = torch.cat((nextRhoHx, thisRhoHX), dim=1)
    #             nextRhoHx[:, nextRhoHx_count: nextRhoHx_count + self.J, :, :, :] = thisRhoHX
    #             nextRhoHx_count += self.J
    #             total_tree_flags += [1] * self.J
    #         else:
    #             total_tree_flags += [0] * self.J

    #     count = 0
    #     leaves_energy = np.zeros(len(total_tree_flags))
    #     for i in range(len(total_tree_flags)):
    #         if total_tree_flags[i] == 1:
    #             leaves_energy[i] = (torch.norm(nextRhoHx[:, count, :, :, :]) ** 2).item()
    #             count += 1
    #     assert len(total_tree_flags) == np.int(np.sum(self.J ** np.arange(0, target_layer+1)))
    #     assert count == nextRhoHx.shape[1]
    #     return leaves_energy, total_tree_flags

    # def computePrunedTransformLeavesEnergyMemLessNoCat(self, torch_x, target_layer, parents_flags):
    #     # Require less memory than previous implementation
    #     # dimension of torch_x: batchSize x F(3) x N_time x N_space
    #     # This part is reserved for Pruned version of this work. Can ignore this part for now.
    #     # See Ioannidis, Vassilis N., Siheng Chen, and Georgios B. Giannakis.
    #     # "Pruned graph scattering transforms." International Conference on Learning Representations. 2019.
    #     assert len(torch_x.shape) == 4
    #     B = torch_x.shape[0]
    #     F = torch_x.shape[1]
    #     # compute how many nodes will be preserved
    #     valid_node_count = sum(parents_flags)

    #     nextRhoHx = torch.empty([B, valid_node_count, F, self.N_time, self.N_space]).cuda()  # at cuda
    #     nextRhoHx[:, 0, :, :, :] = torch_x
    #     nextRhoHx_count = 1
    #     total_tree_flags = [1]
    #     leaves_energy = [torch.norm(torch_x).item()]

    #     # layer 1 to target_layer - 1
    #     for l in range(1, target_layer):
    #         for j in range(self.J ** l):
    #             # B x F x N_time x N_space, at cuda
    #             # this is the signal at their parent node
    #             cur_node_total_tree_index = compute_tree_idx(self.J, l, j)
    #             parent_total_tree_index, _ = find_parent_children(cur_node_total_tree_index, self.J, target_layer+1)
    #             if parents_flags[parent_total_tree_index] == 1 and parents_flags[cur_node_total_tree_index] == 1:
    #                 nextRhoHx_parent_index = int(sum(parents_flags[0: parent_total_tree_index]))
    #                 parentX = nextRhoHx[:, nextRhoHx_parent_index, :, :, :]  # B x F x N_time x N_space
    #                 thisRhoHX = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 space_scale_itr = math.floor((j % self.J) / self.J_time)
    #                 time_scale_itr = (j % self.J) % self.J_time

    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = parentX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr]  # B x F x N_time
    #                     RhoHX_thisJoint = torch.abs(torch.matmul(gstX_thisJoint, self.H_time[time_scale_itr, :, :]))
    #                     thisRhoHX[:, :, :, space_itr] = RhoHX_thisJoint

    #                 # nextRhoHx = torch.cat((nextRhoHx, thisRhoHX), dim=1)
    #                 nextRhoHx[:, nextRhoHx_count, :, :, :] = thisRhoHX
    #                 nextRhoHx_count += 1

    #                 leaves_energy.append(torch.norm(thisRhoHX).item())
    #                 total_tree_flags.append(1)
    #             else:
    #                 leaves_energy.append(0.0)
    #                 total_tree_flags.append(0)

    #     # target_layer
    #     l = target_layer
    #     for j in range(0, self.J ** l, self.J):
    #         cur_node_total_tree_index = compute_tree_idx(self.J, l, j)
    #         parent_total_tree_index, _ = find_parent_children(cur_node_total_tree_index, self.J, target_layer+1)
    #         if parents_flags[parent_total_tree_index] == 1:
    #             nextRhoHx_parent_index = int(sum(parents_flags[0: parent_total_tree_index]))
    #             parentX = nextRhoHx[:, nextRhoHx_parent_index, :, :, :]  # B x F x N_time x N_space
    #             thisRhoHX = torch.empty([B, self.J, F, self.N_time, self.N_space]).cuda()
    #             for space_scale_itr in range(self.J_space):
    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = parentX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr].reshape((B, 1, F, self.N_time))
    #                     RhoHX_thisJoint = torch.abs(torch.matmul(gstX_thisJoint, self.H_time))
    #                     thisRhoHX[:, space_scale_itr*self.J_time: (space_scale_itr+1)*self.J_time,
    #                               :, :, space_itr] = RhoHX_thisJoint

    #             for leaf_order in range(self.J):
    #                 leaves_energy.append(torch.norm(thisRhoHX[:, leaf_order, :, :, :]).item())
    #                 total_tree_flags.append(1)
    #         else:
    #             leaves_energy += [0.0] * self.J
    #             total_tree_flags += [0] * self.J

    #     assert len(total_tree_flags) == np.int(np.sum(self.J ** np.arange(0, target_layer+1)))
    #     assert len(total_tree_flags) == len(leaves_energy)
    #     return np.array(leaves_energy), total_tree_flags

    # def computePrunedTransformNoCatSpaceNoAvg(self, torch_x, valid_node_idx):
    #     # Compute coefficients based on pruned scattering trees. Only averaged over temporal domain
    #     # dimension of torch_x: batchSize x F(3) x N_time x N_space
    #     # This part is reserved for Pruned version of this work. Can ignore this part for now.
    #     # See Ioannidis, Vassilis N., Siheng Chen, and Georgios B. Giannakis.
    #     # "Pruned graph scattering transforms." International Conference on Learning Representations. 2019.
    #     assert len(torch_x.shape) == 4
    #     B = torch_x.shape[0]
    #     F = torch_x.shape[1]
    #     nextRhoHx = torch.empty([B, len(valid_node_idx), F, self.N_time, self.N_space]).cuda()  # at cuda
    #     nextRhoHx[:, 0, :, :, :] = torch_x
    #     nextRhoHx_count = 1
    #     Phi = torch.empty([B, F, self.N_space * len(valid_node_idx)]).cuda()
    #     Phi[:, :, 0: self.N_space] = torch.mean(torch_x, 2)  # batchSize x F(3) x N_space
    #     Phi_count = self.N_space

    #     # iterate each node over all layers except for the root
    #     for l in range(1, self.L):
    #         for j in range(self.J ** l):
    #             # B x F x N_time x N_space, at cuda
    #             # this is the signal at their parent node
    #             cur_node_total_tree_index = compute_tree_idx(self.J, l, j)
    #             parent_total_tree_index, _ = find_parent_children(cur_node_total_tree_index, self.J, self.L)
    #             if cur_node_total_tree_index in valid_node_idx and parent_total_tree_index in valid_node_idx:
    #                 nextRhoHx_parent_index = valid_node_idx.index(parent_total_tree_index)
    #                 parentX = nextRhoHx[:, nextRhoHx_parent_index, :, :, :]  # B x F x N_time x N_space
    #                 thisRhoHX = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 space_scale_itr = math.floor((j % self.J) / self.J_time)
    #                 time_scale_itr = (j % self.J) % self.J_time

    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = parentX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr]  # B x F x N_time
    #                     RhoHX_thisJoint = torch.abs(torch.matmul(gstX_thisJoint, self.H_time[time_scale_itr, :, :]))
    #                     thisRhoHX[:, :, :, space_itr] = RhoHX_thisJoint

    #                 nextRhoHx[:, nextRhoHx_count, :, :, :] = thisRhoHX
    #                 nextRhoHx_count += 1
    #                 Phi[:, :, Phi_count: Phi_count + self.N_space] = torch.mean(thisRhoHX, 2)
    #                 Phi_count += self.N_space

    #     return Phi.cpu().numpy()

    # def computePrunedTransformNoCatNoAvg(self, torch_x, valid_node_idx):
    #     # Compute coefficients based on pruned scattering trees. No pooling is performed
    #     # dimension of torch_x: batchSize x F(3) x N_time x N_space
    #     # This part is reserved for Pruned version of this work. Can ignore this part for now.
    #     # See Ioannidis, Vassilis N., Siheng Chen, and Georgios B. Giannakis.
    #     # "Pruned graph scattering transforms." International Conference on Learning Representations. 2019.
    #     assert len(torch_x.shape) == 4
    #     B = torch_x.shape[0]
    #     F = torch_x.shape[1]
    #     nextRhoHx = torch.empty([B, len(valid_node_idx), F, self.N_time, self.N_space]).cuda()  # at cuda
    #     nextRhoHx[:, 0, :, :, :] = torch_x
    #     nextRhoHx_count = 1

    #     # iterate each node over all layers except for the root
    #     for l in range(1, self.L):
    #         for j in range(self.J ** l):
    #             # B x F x N_time x N_space, at cuda
    #             # this is the signal at their parent node
    #             cur_node_total_tree_index = compute_tree_idx(self.J, l, j)
    #             parent_total_tree_index, _ = find_parent_children(cur_node_total_tree_index, self.J, self.L)
    #             if cur_node_total_tree_index in valid_node_idx and parent_total_tree_index in valid_node_idx:
    #                 nextRhoHx_parent_index = valid_node_idx.index(parent_total_tree_index)
    #                 parentX = nextRhoHx[:, nextRhoHx_parent_index, :, :, :]  # B x F x N_time x N_space
    #                 thisRhoHX = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 space_scale_itr = math.floor((j % self.J) / self.J_time)
    #                 time_scale_itr = (j % self.J) % self.J_time

    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = parentX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr]  # B x F x N_time
    #                     RhoHX_thisJoint = torch.abs(torch.matmul(gstX_thisJoint, self.H_time[time_scale_itr, :, :]))
    #                     thisRhoHX[:, :, :, space_itr] = RhoHX_thisJoint

    #                 nextRhoHx[:, nextRhoHx_count, :, :, :] = thisRhoHX
    #                 nextRhoHx_count += 1

    #     return nextRhoHx.transpose(1, 2).reshape(B, F, -1).cpu().numpy()

    # def computePrunedTransformNoAbs(self, torch_x, valid_node_idx):
    #     # Compute coefficients based on pruned scattering trees. No nonlinearity is used
    #     # Use torch.cat here. Pretty slow
    #     # dimension of torch_x: batchSize x F(3) x N_time x N_space
    #     # This part is reserved for Pruned version of this work. Can ignore this part for now.
    #     # See Ioannidis, Vassilis N., Siheng Chen, and Georgios B. Giannakis.
    #     # "Pruned graph scattering transforms." International Conference on Learning Representations. 2019.
    #     assert len(torch_x.shape) == 4
    #     B = torch_x.shape[0]
    #     F = torch_x.shape[1]
    #     nextRhoHx = torch.empty([B, 1, F, self.N_time, self.N_space]).cuda()  # at cuda
    #     nextRhoHx[:, 0, :, :, :] = torch_x
    #     Phi = torch.mean(torch_x, 2)  # batchSize x F(3) x N_space

    #     # layer 1 to target_layer - 1
    #     for l in range(1, self.L):
    #         for j in range(self.J ** l):
    #             # B x F x N_time x N_space, at cuda
    #             # this is the signal at their parent node
    #             cur_node_total_tree_index = compute_tree_idx(self.J, l, j)
    #             parent_total_tree_index, _ = find_parent_children(cur_node_total_tree_index, self.J, self.L)
    #             if cur_node_total_tree_index in valid_node_idx and parent_total_tree_index in valid_node_idx:
    #                 nextRhoHx_parent_index = valid_node_idx.index(parent_total_tree_index)
    #                 parentX = nextRhoHx[:, nextRhoHx_parent_index, :, :, :]  # B x F x N_time x N_space
    #                 thisRhoHX = torch.empty([B, 1, F, self.N_time, self.N_space]).cuda()
    #                 space_scale_itr = math.floor((j % self.J) / self.J_time)
    #                 time_scale_itr = (j % self.J) % self.J_time

    #                 thisX_after_spatial_gst = torch.empty([B, F, self.N_time, self.N_space]).cuda()
    #                 for time_itr in range(self.N_time):
    #                     thisX_thisTime = parentX[:, :, time_itr, :]  # B x F x N_space
    #                     thisX_after_spatial_gst[:, :, time_itr, :] = torch.matmul(thisX_thisTime,
    #                                                                               self.H_space[space_scale_itr, :, :])
    #                 # sub-layer after spatial gst
    #                 for space_itr in range(self.N_space):
    #                     gstX_thisJoint = thisX_after_spatial_gst[:, :, :, space_itr]  # B x F x N_time
    #                     RhoHX_thisJoint = torch.matmul(gstX_thisJoint, self.H_time[time_scale_itr, :, :])
    #                     thisRhoHX[:, 0, :, :, space_itr] = RhoHX_thisJoint

    #                     phi_node = torch.mean(RhoHX_thisJoint, 2).reshape((B, F, 1))  # B x F x 1
    #                     Phi = torch.cat((Phi, phi_node), dim=2)

    #                 nextRhoHx = torch.cat((nextRhoHx, thisRhoHX), dim=1)

    #     assert Phi.shape[2] == len(valid_node_idx) * self.N_space
    #     return Phi.cpu().numpy()