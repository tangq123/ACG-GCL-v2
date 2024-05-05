#  Our datasets are fetched from dgl 0.4.3 version.
# from dgl.data import CoraDataset, CitationGraphDataset,AmazonCoBuy, Coauthor
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os
import torch
from torch.utils.data import random_split

def normalize_adj_torch(adj,self_loop=False):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = torch.eye(adj.shape[0]).to(adj.device) + adj
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to(adj.device)
    return torch.mm(d_mat_inv_sqrt,torch.mm(adj,d_mat_inv_sqrt))

def normalize_adj_torch_sparse(adj_matrix,self_loop=False):

    if self_loop:
        # Step 1: 添加自环
        num_nodes = adj_matrix.size(0)
        self_loop_values = torch.ones(num_nodes)
        self_loop_indices = torch.arange(num_nodes)
        self_loop_adj_matrix = torch.sparse.FloatTensor(
            torch.stack([self_loop_indices, self_loop_indices]),
            self_loop_values,
            torch.Size([num_nodes, num_nodes])
        )
        adj_matrix = adj_matrix + self_loop_adj_matrix.to(adj_matrix.device)

    # Step 2: 计算 (A+I) 的度矩阵 D
    degree_AplusI = torch.sparse.sum(adj_matrix, dim=1).to_dense()
    # 防止节点度数为0的情况
    degree_AplusI[degree_AplusI == 0] = 1

    # Step 3: 构建自环对称归一化邻接矩阵
    # D^(-0.5)
    degree_sqrt_inv = torch.pow(degree_AplusI, -0.5)
    # 对角矩阵D^(-0.5)
    D_sqrt_inv = torch.diag(degree_sqrt_inv.squeeze()).to(adj_matrix.device)

    # A' = (D)^(-0.5) * (A+I) * (D)^(-0.5)
    normalized_adj_matrix = torch.sparse.mm(D_sqrt_inv,torch.sparse.mm(adj_matrix,D_sqrt_inv))

    return normalized_adj_matrix.to_dense()

def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)
    
def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    idx_train, idx_val, idx_test = random_split(torch.arange(0, num_samples), (train_len, val_len, test_len))
    print(idx_train)
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)

def download(dataset):
    if dataset == 'cora':
        return CoraDataset()
    elif dataset == 'citeseer' or dataset =='pubmed':
        return CitationGraphDataset(name=dataset)
    elif dataset == 'computers' or dataset =='photo':
        return AmazonCoBuy(name=dataset)
    elif dataset == 'cs' or dataset =='physics':
        return Coauthor(name=dataset)
    else:
        return None


def load(dataset):
    datadir = os.path.join('data', dataset)

    if not os.path.exists(datadir):
        os.makedirs(datadir)
        ds = download(dataset)
        print(type(ds))
        if dataset in ['cora','citeseer','pubmed']:
            adj = nx.to_numpy_array(ds.graph)
            feat = ds.features[:]
            labels = ds.labels[:]
        elif dataset in ['computers', 'photo','cs','physics']:
            ds = ds[0]
            adj = nx.to_numpy_array(ds.to_networkx())
            feat = ds.ndata['feat'] 
            labels = ds.ndata['label']     
        if dataset in ['cora','citeseer','pubmed']:
            idx_train = np.argwhere(ds.train_mask == 1).reshape(-1)
            idx_val = np.argwhere(ds.val_mask == 1).reshape(-1)
            idx_test = np.argwhere(ds.test_mask == 1).reshape(-1)
        elif dataset in ['computers', 'photo','cs','physics']:
            idx_train, idx_val, idx_test = generate_split(ds.number_of_nodes(),0.1,0.1)
        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)
        np.save(f'{datadir}/idx_train.npy', idx_train)
        np.save(f'{datadir}/idx_val.npy', idx_val)
        np.save(f'{datadir}/idx_test.npy', idx_test)
    else:
        adj = np.load(f'{datadir}/adj.npy')
        feat = np.load(f'{datadir}/feat.npy')
        labels = np.load(f'{datadir}/labels.npy')
        idx_train = np.load(f'{datadir}/idx_train.npy')
        idx_val = np.load(f'{datadir}/idx_val.npy')
        idx_test = np.load(f'{datadir}/idx_test.npy')

    # if dataset == 'citeseer':
    #     print('processing')
    #     feat = preprocess_features(feat)
    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    return adj, feat, labels, idx_train, idx_val, idx_test

def load_Cora_AmazonComputer(dataset):

    datadir = os.path.join('data', dataset)
    adj = torch.load(f'{datadir}/adj.pt')
    feat = torch.load(f'{datadir}/feat.pt').numpy()
    labels = torch.load(f'{datadir}/labels.pt').numpy()
    idx_train = torch.load(f'{datadir}/idx_train.pt')
    idx_val = torch.load(f'{datadir}/idx_val.pt')
    idx_test = torch.load(f'{datadir}/idx_test.pt')


    #adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    adj = normalize_adj_torch(adj,self_loop=True).numpy()
    return adj, feat, labels, idx_train, idx_val, idx_test


if __name__ == '__main__':
    # 'cora', 'citeseer', 'pubmed','photo','cs','computers','physics'
    for dataset in ['cora', 'citeseer', 'pubmed','computers', 'photo','cs','physics']:
        load(dataset)
