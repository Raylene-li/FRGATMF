'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import random

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


set_seed(42)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# def preprocess_graph(adj):
#     adj_dense = adj.to_dense()  # 转换为稠密张量
#     adj_np = adj_dense.numpy()  # 转换为NumPy数组
#     adj = sp.coo_matrix(adj_np)
#     adj_ = adj + sp.eye(adj.shape[0])
#     rowsum = np.array(adj_.sum(1))
#     degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
#     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#     return sparse_to_tuple(adj_normalized)


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def mask_test_edges(adj):  # 训练集：验证集：测试集=7：2：1
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    # adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # 保留上三角区域，避免边重复
    adj_tuple = sparse_to_tuple(adj_triu)  # 稀疏矩阵变为元组
    edges = adj_tuple[0]  # 获取边列表 1218,2
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges_0 = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    # 互换起始节点和终止节点，得到反向边
    reversed_edges = np.flip(train_edges_0, axis=0)
    # 将反向边添加到原始边后面，得到扩展后的边
    train_edges = np.concatenate([train_edges_0, reversed_edges], axis=0)  # 2074,2
    # print("train_edges")
    # print(train_edges.shape)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])


    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(train_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)


    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    # adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # adj_train = adj_train + adj_train.T

    # 创建 train 数据对应的稀疏张量
    rows = torch.tensor(train_edges[:, 0], dtype=torch.long)
    cols = torch.tensor(train_edges[:, 1], dtype=torch.long)
    values = torch.tensor(data, dtype=torch.float)
    train_shape = adj.shape

    adj_train = torch.sparse.FloatTensor(torch.stack([rows, cols]), values, train_shape)

    # 对称化
    adj_train = adj_train + adj_train.t()  # Tensor,layout=torch.sparse_coo  边数2074
    # print("adj_train")
    # print(type(adj_train))
    # print(adj_train)
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def mask_test_edges2(adj):  # 训练集：测试集=8：2
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    # adj.eliminate_zeros()
    # # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # 保留上三角区域，避免边重复
    adj_tuple = sparse_to_tuple(adj_triu)  # 稀疏矩阵变为元组
    edges = adj_tuple[0]  # 获取边列表 1218,2
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] * 0.2))  # 1 2 3 4 5 6 7 8 9
    # print("edges.shape[0]")
    # print(edges.shape[0])
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    test_edge_idx = all_edge_idx[:num_test]
    test_edges = edges[test_edge_idx]
    train_edges_0 = np.delete(edges, test_edge_idx, axis=0)  # 975
    # 互换起始节点和终止节点，得到反向边
    reversed_edges = np.flip(train_edges_0, axis=0)
    # 将反向边添加到原始边后面，得到扩展后的边
    train_edges = np.concatenate([train_edges_0, reversed_edges], axis=0)  # 2074,2

    # 构建训练集和测试集的负样本
    edges_graph = nx.from_edgelist(edges, create_using=nx.Graph())  # 创建无向图
    edges_false_a = list(nx.non_edges(edges_graph))
    np.random.shuffle(edges_false_a)
    test_edges_false = edges_false_a[:num_test]
    num_train = len(train_edges_0) + num_test
    train_edges_false = edges_false_a[num_test:num_train]  # 训练集正样本和负样本个数一致
    # print("train_edges")
    # print(train_edges)
    train_shape = adj.shape
    data = []
    for edge in train_edges:
        edge_index = np.where((adj_tuple[0][:, 0] == edge[0]) & (adj_tuple[0][:, 1] == edge[1]))[0][0]
        weight = adj_tuple[1][edge_index]
        data.append(weight)
    data = np.array(data)

    # adj = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    # rows, cols = adj.nonzero()
    # adj = torch.sparse_coo_tensor(torch.stack([torch.tensor(rows), torch.tensor(cols)]), torch.tensor(adj.data), train_shape)

    # print("train_edges")
    # print(len(train_edges))
    # print("train_edges_false")
    # print(len(train_edges_false))
    # print("test_edges")
    # print(len(test_edges))
    # print("test_edges_false")
    # print(len(test_edges_false))

    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)  # gae
    # rows, cols = adj.nonzero()  # 转为稀疏张量 gcn-svd
    # adj = torch.sparse_coo_tensor(torch.stack([torch.tensor(rows), torch.tensor(cols)]), torch.tensor(adj.data),
    #                               adj.shape)
    # adj_train = adj_train + adj_train.T
    return adj_train, train_edges, train_edges_false, test_edges, test_edges_false
