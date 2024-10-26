import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from torch.nn.parameter import Parameter
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from utils import get_scores
from torch_geometric.nn import GATConv
from torch_geometric.nn import InnerProductDecoder
import math
import args

class geo_GAT2(torch.nn.Module):
    def __init__(self, in_channels, nhid, out_channels, heads):  # in_channels:120
        super(geo_GAT2, self).__init__()
        #　1层
        # self.conv1 = GATConv(in_channels, out_channels, concat=False)
        # 2层
        self.conv1 = GATConv(in_channels, nhid, heads=heads)
        # 调整输入通道数，因为多头注意力的输出是头数乘以每头的输出维度,二层
        self.conv2 = GATConv(nhid * heads, out_channels, concat=False)
        self.decoder = InnerProductDecoder()
        self.relu = nn.LeakyReLU(0.2)
        self.res1 = torch.nn.Linear(in_channels, nhid * heads)
        self.res2 = torch.nn.Linear(nhid * heads, nhid * heads)
        self.fc = torch.nn.Linear(nhid * heads, out_channels)

    def encode(self, x, edge_index):  # edge_index:(2,2074)
        # residual gat
        x_res = self.res1(x)  # 2层
        x = self.conv1(x, edge_index) + x_res
        x_res = self.fc(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index) + x_res
        x = self.relu(x)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        # 解码步骤保持不变，使用内积来计算边的得分
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


class DANMF(object):
    """
    Deep autoencoder-like non-negative matrix factorization class.
    """
    def __init__(self, graph, args, adj2):
        super(DANMF, self).__init__()
        """
        Initializing a DANMF object.
        :param graph: Networkx graph.
        :param args: Arguments object.
        """
        self.graph = graph
        self.A = nx.adjacency_matrix(self.graph)  # 边数2436
        self.L = nx.laplacian_matrix(self.graph)
        self.D = self.L+self.A
        self.args = args
        self.p = len(self.args.layers)
        self.A = self.A + adj2

        # 归一化
        min_val = min(self.A.data)
        max_val = max(self.A.data)
        self.A.data = (self.A.data - min_val) / (max_val - min_val)

        t = args.threshold
        self.A.data[self.A.data < t] = 0
        self.A.data[self.A.data >= t] = 1

        self.lamb = 0.1

        self.l1_lambda = 1e-4
        self.l2_lambda = 1e-2
        self.A_ori = self.A


    def setup_z(self, i):  # 邻接矩阵
        """
        Setup target matrix for pre-training process.
        """
        if i == 0:
            self.Z = self.A
        else:
            self.Z = self.V_s[i-1]


    def sklearn_pretrain(self, i):
        """  NMF（非负矩阵分解）模型预训练模型的单层
        Pretraining a single layer of the model with sklearn.
        :param i: Layer index.
        n_components：指定分解后的特征数量
        """
        nmf_model = NMF(n_components=self.args.layers[i],
                        init="random",
                        random_state=self.args.seed,
                        max_iter=self.args.pre_iterations)
        U = nmf_model.fit_transform(self.Z)  # (120, 120)、(120, 32) 系数矩阵=权重矩阵 行不变
        V = nmf_model.components_  # (120, 120)、(32, 120) 基矩阵 列不变
        return U, V

    def pre_training(self):
        """
        Pre-training each NMF layer.
        """
        print("\nLayer pre-training started. \n")
        # self.A = self.gc1(pre_emb, self.A)
        self.U_s = []
        self.V_s = []
        for i in tqdm(range(self.p), desc="Layers trained: ", leave=True):
            self.setup_z(i)
            U, V = self.sklearn_pretrain(i)
            self.U_s.append(U)  # 存储每层的系数矩阵
            self.V_s.append(V)  # 存储每层的基矩阵

    def setup_Q(self):
        """
        Setting up Q matrices.
        存储两组因子的中间结果，交替更新 U 和 V 时用到 Q 矩阵
        长度为 p (迭代次数)+1
        """
        self.Q_s = [None for _ in range(self.p+1)]
        self.Q_s[self.p] = np.eye(self.args.layers[self.p-1])  # 最后一个元素初始化为单位矩阵：Q_p=I
        for i in range(self.p-1, -1, -1):  # 从倒数第二个元素开始，逐步计算并存储其他层 Q 矩阵的值
            self.Q_s[i] = np.dot(self.U_s[i], self.Q_s[i+1])  # 已知的系数矩阵 * 下一层 Q 矩阵的值

    def update_U(self, i):
        """
        Updating left hand factors.
        :param i: Layer index.
        Φ^T_{𝑖+1} -> Q_s  Ψ^𝑇_{𝑖−1} -> 系数矩阵 U_s
        """
        if i == 0:
            R = self.U_s[0].dot(self.Q_s[1].dot(self.VpVpT).dot(self.Q_s[1].T))
            R = R+self.A_sq.dot(self.U_s[0].dot(self.Q_s[1].dot(self.Q_s[1].T)))
            Ru = 2*self.A.dot(self.V_s[self.p-1].T.dot(self.Q_s[1].T))  # 分子
            self.U_s[0] = (self.U_s[0]*Ru)/np.maximum(R, 10**-10)  # 使用np.maximum确保分母不为0
        else:
            R = self.P.T.dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.VpVpT).dot(self.Q_s[i+1].T)
            R = R+self.A_sq.dot(self.P).T.dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.Q_s[i+1].T)
            Ru = 2*self.A.dot(self.P).T.dot(self.V_s[self.p-1].T).dot(self.Q_s[i+1].T)
            self.U_s[i] = (self.U_s[i]*Ru)/np.maximum(R, 10**-10)

    def update_P(self, i):
        """
        Setting up P matrices.
        :param i: Layer index.
        """
        if i == 0:
            self.P = self.U_s[0]
        else:
            self.P = self.P.dot(self.U_s[i])

    def update_V(self, i):
        """
        Updating right hand factors.
        :param i: Layer index.
        """
        if i < self.p-1:  # V_i
            Vu = 2*self.A.dot(self.P).T
            Vd = self.P.T.dot(self.P).dot(self.V_s[i])+self.V_s[i]
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd, 10**-10)
        else:  # V_p
            Vu = 2*self.A.dot(self.P).T+(self.args.lamb*self.A.dot(self.V_s[i].T)).T
            Vd = self.P.T.dot(self.P).dot(self.V_s[i])
            Vd = Vd + self.V_s[i]+(self.args.lamb*self.D.dot(self.V_s[i].T)).T
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd, 10**-10)

    def calculate_cost(self, i):
        """
        Calculate loss.
        :param i: Global iteration.重构损失和正则化损失
        """
        reconstruction_loss_1 = np.linalg.norm(self.A-self.P.dot(self.V_s[-1]), ord="fro")**2
        reconstruction_loss_2 = np.linalg.norm(self.V_s[-1]-self.A.dot(self.P).T, ord="fro")**2
        regularization_loss = np.trace(self.V_s[-1].dot(self.L.dot(self.V_s[-1].T)))
        self.loss.append([i+1, reconstruction_loss_1, reconstruction_loss_2, regularization_loss])


    def save_embedding(self):
        """
        Save embedding matrix.
        """
        embedding = [np.array(range(self.P.shape[0])).reshape(-1, 1), self.P, self.V_s[-1].T]
        embedding = np.concatenate(embedding, axis=1)
        columns = ["id"] + ["x_" + str(x) for x in range(self.args.layers[-1]*2)]
        return embedding

    def save_membership(self):
        """
        Save cluster membership.
        """
        index = np.argmax(self.P, axis=1)
        self.membership = {int(i): int(index[i]) for i in range(len(index))}
        with open(self.args.membership_path, "w") as f:
            json.dump(self.membership, f)

    def get_acc(self, adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy

    def training(self):
        """
        Training process after pre-training.
        """
        print("\n\nTraining started. \n")
        self.loss = []
        self.A_sq = self.A.dot(self.A.T)  # 计算协方差矩阵，增强图的表示，提供节点间的相似性度量
        for iteration in tqdm(range(self.args.iterations), desc="Training pass: ", leave=True):
            self.setup_Q()  # 设定中间矩阵
            self.VpVpT = self.V_s[self.p-1].dot(self.V_s[self.p-1].T)  # 计算基矩阵内积或相似度
            for i in range(self.p):  # 逐层循环
                self.update_U(i)
                self.update_P(i)
                self.update_V(i)

            if self.args.calculate_loss:
                self.calculate_cost(iteration)
                # 下载评估结果最好的emb

        emb = self.save_embedding()
        S = self.U_s[0].dot(self.U_s[1]).dot(self.V_s[1])
        return S

