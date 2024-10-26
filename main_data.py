import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import scipy.sparse as sp
import numpy as np
import os
import time
import glob

from input_data import load_data, load_mydata, read_graph, load_mydata_gat, load_data3
from preprocessing import *
from torch import optim, nn
import args
from model import *
from utils import *
from args import parameter_parser
from torch.autograd import Variable


args = parameter_parser()


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


set_seed(args.seed)
random.seed(args.seed)


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

adj, graph, features = load_mydata(args.num_node)  # # Acupuncture

# adj, features, graph = load_data("cora")   # cora citeseer
# adj, features, graph = load_data3(args.num_node)  # winconsin
features = Variable(features)

# Store original adjacency matrix (without diagonal entries) for later  无自环邻接矩阵
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)  # 去除自环
adj_orig.eliminate_zeros()

adj_train, train_edges, train_edges_false, test_edges, test_edges_false = mask_test_edges2(adj)

train_edges = torch.tensor(train_edges, dtype=torch.long)
train_edges = train_edges.transpose(0, 1)  # 转换维度为(2,1037)，用于输入gat

train_edges_false = torch.tensor(train_edges_false, dtype=torch.long)
train_edges_false = train_edges_false.transpose(0, 1)  # 转换维度为(2,1037)，用于输入gat

test_edges = torch.tensor(test_edges, dtype=torch.long)
test_edges = test_edges.transpose(0, 1)  # 转换维度为(2,1037)，用于输入gat

test_edges_false = torch.tensor(test_edges_false, dtype=torch.long)
test_edges_false = test_edges_false.transpose(0, 1)  # 转换维度为(2,1037)，用于输入gat


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train_gat(re_adj, args):
    re_adj = torch.tensor(re_adj, dtype=torch.float)
    model = geo_GAT2(in_channels=re_adj.shape[1],
                    nhid=args.hid_features,
                    out_channels=args.sec_hidden,
                    heads=args.nb_heads)

    optimizer = optim.Adam(model.parameters(), lr=args.gat_lr, weight_decay=args.gat_weight_decay)
    loss_values = []
    best = 1e9
    cnt_wait = 0
    lr_auc = []
    lr_ap = []

    for epoch in range(args.gat_epoches):
        start_time_iter = time.time()
        model.train()
        optimizer.zero_grad()
        z = model.encode(re_adj, train_edges)  # 节点嵌入 120,64  训练集正样本：975*2 = 2316（包括正反）
        link_logits = model.decode(z, train_edges, train_edges_false)
        link_labels = get_link_labels(train_edges, train_edges_false)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss_values.append(loss.tolist())

        if epoch % 5 == 0:
            test_link_logits = model.decode(z, test_edges, test_edges_false)
            test_link_probs = test_link_logits.sigmoid()
            test_link_probs = test_link_probs.detach().numpy()  # 使用 detach() 方法将其转换为不需要梯度的张量，然后再转换为 NumPy 数组
            test_link_labels = get_link_labels(test_edges, test_edges_false)
            auc = roc_auc_score(test_link_labels, test_link_probs)  # 获取每五个epoch
            ap = average_precision_score(test_link_labels, test_link_probs)
            lr_auc.append(auc)
            lr_ap.append(ap)

        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 更新所有参数
        end_time_iter = time.time()
        if loss < best:
            best = loss
            cnt_wait = 0
            # torch.save(model.state_dict(), 'best_gat.pth')
            torch.save(z, 'best.pth')
        else:
            cnt_wait += 1

        print("Epoch-> {0}  , Iteration_time-> {1:.4f} , train_loss {2:.4f}".format(
            epoch, end_time_iter - start_time_iter, loss.data.item()))

    model.eval()
    z = torch.load('best.pth')
    test_link_logits = model.decode(z, test_edges, test_edges_false)
    test_link_probs = test_link_logits.sigmoid()
    test_link_probs = test_link_probs.detach().numpy()  # 使用 detach() 方法将其转换为不需要梯度的张量，然后再转换为 NumPy 数组
    test_link_labels = get_link_labels(test_edges, test_edges_false)
    test_roc = roc_auc_score(test_link_labels, test_link_probs)
    average_precision = average_precision_score(test_link_labels, test_link_probs)
    print("gat_test_roc:{0} gat_test_ap:{1}".format(test_roc, average_precision))
    return test_roc, average_precision


if __name__ == "__main__":
    node_sim = emb_node2(features)  # 根据节点嵌入得到相似性度量
    mf_model = DANMF(graph, args, node_sim)  # graph cora citeseer win数据集；graph1 mydata，配置，与adj融合的信息
    mf_model.pre_training()
    pred_matrix = mf_model.training()  # 潜在向量
    result = train_gat(pred_matrix, args)  # features
    print(result)
