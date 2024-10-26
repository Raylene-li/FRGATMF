'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import math
import torch
from utils import largest_connected_components

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    # dataset = 'cora'
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    print()
    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.toarray()))  # cora citeseer有  gae无
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # csr格式 2708,2708
    return adj, features, nx.Graph(graph)  # from_dict_of_lists

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)



def load_mydata(node):
    print('Loading {} dataset...'.format("mydata"))
    pre_emb = []  # 特征emb
    with open('mydata/pre_train120.emb', 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            embedding = list(map(float, parts[1:]))
            pre_emb.append(embedding)
    features = sp.csr_matrix(pre_emb, dtype=np.float32)
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))

    # 读取带权重的adj
    edge_list = pd.read_csv('mydata/acu_acu.txt', header=None, sep=' ').values.tolist()
    adj = sp.lil_matrix((node, node))
    rows = []
    lines = []
    values = []
    cnt = 0
    G = nx.Graph()
    for (id1, id2, weight) in edge_list:
        rows.append(int(id1))
        lines.append(int(id2))
        values.append(1)
        G.add_edge(int(id1), int(id2))
    adj = sp.csr_matrix((values, (rows, lines)), shape=(node, node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    return adj, G, features


def load_data3(node):
    # 读取带权重的adj
    edge_list = pd.read_csv('data/wisconsin_edges.txt', header=None, sep='	').values.tolist()
    rows = []
    lines = []
    values = []
    for (id1, id2) in edge_list:
        rows.append(int(id1))
        lines.append(int(id2))
        values.append(1)
    adj = sp.csr_matrix((values, (rows, lines)), shape=(node, node))
    graph = nx.from_scipy_sparse_matrix(adj)

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    # graph = nx.from_scipy_sparse_matrix(adj)

    # print("adj.nnz")
    # print(adj.nnz)
    # load features
    pre_emb = []  # 特征嵌入
    with open('data/wisconsin_feature.emb', 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            embedding = list(map(float, parts[1:]))
            pre_emb.append(embedding)
    features = sp.csr_matrix(pre_emb, dtype=np.float32)
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    # features = torch.FloatTensor(features)
    # graph = nx.from_scipy_sparse_matrix(adj)
    # rows, cols = features.nonzero()  # 转为稀疏张量
    # features = torch.sparse_coo_tensor(torch.stack([torch.tensor(rows), torch.tensor(cols)]), torch.tensor(features.data), features.shape)
    return adj, features, graph


def read_graph():
    """
    Method to read graph and create a target matrix with matrix powers.
    :param args: Arguments object.
    """
    print("\nTarget matrix creation started.\n")
    edge_list = pd.read_csv('mydata/acu_acu.txt', header=None, sep=' ').values.tolist()
    graph = nx.Graph()
    for (id1, id2, weight) in edge_list:
        # print("(id1, id2, weight)")
        # print((id1, id2, weight))
        graph.add_edge(id1, id2, weight=1)
    # graph = nx.from_edgelist(edge_list, create_using=nx.Graph())  # 创建无向图
    return graph


def load_mydata_gat(path="./mydata/", dataset="mydata"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    pre_emb = []  # 特征嵌入
    with open('mydata/pre_train120.emb', 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            embedding = list(map(float, parts[1:]))
            pre_emb.append(embedding)
    features = sp.csr_matrix(pre_emb, dtype=np.float32)
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    return features


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_mydata_gat2(path="./mydata/", dataset="mydata"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    pre_emb = []  # 特征嵌入
    with open('mydata/pre_train120.emb', 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            embedding = list(map(float, parts[1:]))
            pre_emb.append(embedding)
    features = sp.csr_matrix(pre_emb, dtype=np.float32)
    # labels = np.ones(120, dtype=np.int32)  # numpy.ndarray
    labels = pd.read_csv('mydata/acu_id.txt').values
    # print("labels")
    # print(type(labels))
    # print(labels)
    # build graph
    edge_list = pd.read_csv('mydata/acu_acu.txt', header=None, sep=' ').values.tolist()
    edges = np.array(edge_list, dtype=np.int32)
    # 边信息为1
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # adj = torch.FloatTensor(np.array(adj.todense()))
    features = normalize_features(features)

    x_train = math.floor(len(labels) * 0.7)  # 向下取整
    x_val = math.floor(len(labels) * 0.8)
    idx_train = range(x_train)
    idx_val = range(x_train, x_val)
    idx_test = range(x_val, len(labels))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_mydata_gae():
    pre_emb = []  # 特征嵌入
    with open('mydata/pre_train120.emb', 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            embedding = list(map(float, parts[1:]))
            pre_emb.append(embedding)
    features = sp.csr_matrix(pre_emb, dtype=np.float32)
    # labels = np.ones(120, dtype=np.int32)  # numpy.ndarray
    labels = pd.read_csv('mydata/acu_id.txt').values
    # print("labels")
    # print(type(labels))
    # print(labels)
    # build graph
    edge_list = pd.read_csv('mydata/acu_acu.txt', header=None, sep=' ').values.tolist()
    edges = np.array(edge_list, dtype=np.int32)
    # 边信息为1
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # adj = torch.FloatTensor(np.array(adj.todense()))
    features = normalize_features(features)

    x_train = math.floor(len(labels) * 0.7)  # 向下取整
    x_val = math.floor(len(labels) * 0.8)
    idx_train = range(x_train)
    idx_val = range(x_train, x_val)
    idx_test = range(x_val, len(labels))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test



def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
