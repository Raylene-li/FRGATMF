"""Data reading utilities."""

import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from texttable import Texttable
import torch
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    ----------
    adj : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def get_ap(train_nodes, edges_pos, edges_neg, S):
    edges_scores = {}

    for i in edges_pos:
        edges_scores[i] = S[[train_nodes.index(i[0])], [train_nodes.index(i[1])]]

    for j in edges_neg:
        edges_scores[j] = S[[train_nodes.index(j[0])], [train_nodes.index(j[1])]]

    sorted_node_pairs = [key for key, value in sorted(edges_scores.items(), key=lambda x: x[1], reverse=True)]

    predict_pair = sorted_node_pairs[:len(edges_pos)]
    Lr = 0

    for pair in predict_pair:
        if pair in edges_pos:
            Lr += 1
    score = Lr / len(edges_pos)
    return score


def get_scores(edges_pos, edges_neg, adj_rec, graph):
    # def sigmoid(x):
    #     x = np.array(x)
    #     return 1 / (1 + np.exp(-x))
    # # print("edges_pos")
    # # print(edges_pos)
    # # gae、danmf
    # edges_pos = edges_pos.transpose(0, 1)  # 转换维度为(2,1037)，用于输入gat
    # edges_neg = edges_neg.transpose(0, 1)  # 维度为(2,n)转为(n,2)
    #
    # # Predict on test set of edges
    # preds = []
    # pos = []
    # for e in edges_pos:
    #     preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
    #     # preds.append(adj_rec[e[0], e[1]].item())
    #     pos.append(adj_orig[e[0], e[1]])
    #
    # preds_neg = []
    # neg = []
    # for e in edges_neg:
    #     preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
    #     # preds_neg.append(adj_rec[e[0], e[1]].item())
    #     neg.append(adj_orig[e[0], e[1]])
    #
    # preds_all = np.hstack([preds, preds_neg])
    # labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    # roc_score = roc_auc_score(labels_all, preds_all)
    # # print("labels_all")
    # # print(labels_all)
    # ap_score = average_precision_score(labels_all, preds_all)
    # # # 计算准确率
    # # preds_all = (preds_all >= 0.5).astype(int)
    # #
    # # accuracy = accuracy_score(labels_all, preds_all)
    # #
    # # # 计算精确度
    # # precision = precision_score(labels_all, preds_all)
    # #
    # # # 计算召回率
    # # recall = recall_score(labels_all, preds_all)
    # #
    # # # 计算F1值
    # # f1 = f1_score(labels_all, preds_all)
    # #
    # # return roc_score, ap_score, accuracy, precision, recall, f1
    # return roc_score, ap_score, preds_all

    y_true = np.array([], dtype=int)
    y_score = np.array([], dtype=int)
    train_nodes = list(graph.nodes())

    for i in edges_pos:
        y_score = np.append(y_score, adj_rec[[train_nodes.index(i[0])],[train_nodes.index(i[1])]])
        y_true = np.append(y_true,1)

    for i in edges_neg:
        y_score = np.append(y_score,adj_rec[[train_nodes.index(i[0])],[train_nodes.index(i[1])]])
        y_true = np.append(y_true,0)
    # print("y_true")
    # print(y_true)
    # print("y_score")
    # print(y_score)
    auc = roc_auc_score(y_true, y_score)
    ap = get_ap(train_nodes, edges_pos, edges_neg, adj_rec)
    return auc, ap

def get_acc(adj_rec, adj_label):
    # accuracy
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)

    # precision
    true_positives = ((adj_rec > 0.5) & (adj_label.to_dense() == 1)).sum().float()
    # 计算 Predicted Positives（预测正例）
    predicted_positives = (adj_rec > 0.5).sum().float()
    # 计算精确度
    precision = true_positives / predicted_positives if predicted_positives != 0 else 0.0

    # recall
    # 计算 Actual Positives（实际正例）
    actual_positives = (adj_label.to_dense() == 1).sum().float()

    # 计算召回率
    recall = true_positives / actual_positives if actual_positives != 0 else 0.0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0


    return accuracy, precision, recall, f1_score


def loss_printer(losses):
    """
    Printing the losses for each iteration.
    :param losses: List of losses in each iteration.
    """
    t = Texttable()
    t.add_rows([["Iteration",
                 "Reconstrcution Loss I.",
                 "Reconstruction Loss II.",
                 "Regularization Loss"]])
    t.add_rows(losses)
    print(t.draw())


def emb_all():
    pre_emb = []
    with open('mydata/pre_train120.emb', 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            embedding = list(map(float, parts[1:]))
            pre_emb.append(embedding)
    pre_emb = np.array(pre_emb)
    # adjacency_matrix = csr_matrix(pre_emb)
    # adjacency_matrix.data[adjacency_matrix.data < 0] = 0

    # 计算属性之间的余弦相似度
    similarity_matrix = cosine_similarity(pre_emb)
    # print("similarity_matrix")
    # print(similarity_matrix)
    # 基于相似度值构建属性邻接矩阵
    adjacency_matrix = csr_matrix(similarity_matrix)
    # adjacency_matrix.data[adjacency_matrix.data < 0] = 0

    # # 设置相似度阈值，根据阈值确定是否存在边
    # threshold = 0.5  # 可根据具体情况调整
    # adjacency_matrix = np.where(similarity_matrix > threshold, 1, 0)
    # adjacency_matrix = torch.from_numpy(adjacency_matrix)
    # adjacency_matrix = csr_matrix(adjacency_matrix)

    # mf_emb = np.array(mf_emb)
    # com_emb = np.hstack([mf_emb, pre_emb])
    # com_emb = torch.from_numpy(com_emb)
    # return com_emb
    return adjacency_matrix

def result_ind(test_real_outputs, test_fake_outputs):
    # 计算均方误差
    mse = mean_squared_error(test_real_outputs.detach().numpy(), test_fake_outputs.detach().numpy())

    # 将输出转换为二元分类结果（例如通过阈值或者sigmoid函数）
    test_real_preds = (test_real_outputs > 0).float().detach().numpy()
    test_fake_preds = (test_fake_outputs > 0).float().detach().numpy()

    # 计算准确率
    accuracy = accuracy_score(np.concatenate([np.ones_like(test_real_preds), np.zeros_like(test_fake_preds)]),
                              np.concatenate([test_real_preds, test_fake_preds]))

    # 计算精确率
    precision = precision_score(np.concatenate([np.ones_like(test_real_preds), np.zeros_like(test_fake_preds)]),
                                np.concatenate([test_real_preds, test_fake_preds]), zero_division='warn')

    # 计算召回率
    recall = recall_score(np.concatenate([np.ones_like(test_real_preds), np.zeros_like(test_fake_preds)]),
                          np.concatenate([test_real_preds, test_fake_preds]))

    # 计算F1-score
    f1 = f1_score(np.concatenate([np.ones_like(test_real_preds), np.zeros_like(test_fake_preds)]),
                  np.concatenate([test_real_preds, test_fake_preds]))
    return mse, accuracy, precision, recall, f1


class GraphDataset(Dataset):
    def __init__(self, edges, node_embeddings):
        super(GraphDataset, self).__init__()
        self.edges = edges
        self.node_embeddings = node_embeddings

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        edge = self.edges[index]
        source_node_embed = self.node_embeddings[int(edge[0])]
        target_node_embed = self.node_embeddings[int(edge[1])]
        return torch.Tensor(source_node_embed), torch.Tensor(target_node_embed), int(edge[0]), int(edge[1])


def accuracy(output, labels):
    # print("output")
    # print(output)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def emb_node():
    # emb = torch.load('gat_emb.pt')
    # emb_np = emb.detach().numpy()
    # similarity_matrix = cosine_similarity(emb_np)
    # adjacency_matrix = csr_matrix(similarity_matrix)
    # return adjacency_matrix
    # pre_emb = []
    # with open('mydata/pre_train120.emb', 'r') as f:
    #     lines = f.readlines()[1:]
    #     for line in lines:
    #         parts = line.strip().split()
    #         embedding = list(map(float, parts[1:]))
    #         pre_emb.append(embedding)
    # pre_emb = np.array(pre_emb)
    similarity_matrix = cosine_similarity(pre_emb)
    adjacency_matrix = csr_matrix(similarity_matrix)
    return adjacency_matrix


def emb_node3():
    pre_emb = []
    with open('mydata/pre_train120.emb', 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            embedding = list(map(float, parts[1:]))
            pre_emb.append(embedding)
    pre_emb = np.array(pre_emb)
    similarity_matrix = cosine_similarity(pre_emb)
    similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float)
    return similarity_matrix


def emb_node2(features):
    # pre_emb = []
    # with open('mydata/pre_train120.emb', 'r') as f:
    #     lines = f.readlines()[1:]
    #     for line in lines:
    #         parts = line.strip().split()
    #         embedding = list(map(float, parts[1:]))
    #         pre_emb.append(embedding)
    # pre_emb = np.array(pre_emb)
    # print("pre_emb")
    # print(type(pre_emb))
    # print(pre_emb)
    similarity_matrix = cosine_similarity(features)
    # sim_data = pd.DataFrame(similarity_matrix)
    # sim_data.to_excel("sim_data.xlsx")
    # print("similarity_matrix")
    # print(type(similarity_matrix))
    adjacency_matrix = csr_matrix(similarity_matrix)  # 2708,2708
    # print("adjacency_matrix")
    # print(adjacency_matrix)
    return adjacency_matrix

