# from load_data import *
import math
import random
import warnings
from MODEL import ViewLearner, MVGRL, LogReg
import pandas as pd

warnings.filterwarnings('ignore')
import torch
from dataset import load, normalize_adj_torch
from utils import setup_seed, eva, target_distribution
from torch.optim import Adam
import torch.nn.functional as F
# import dgl
from sklearn.cluster import KMeans
import torch.nn as nn
import numpy as np

warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import fractional_matrix_power, inv
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
# import seaborn as sns
import matplotlib

# matplotlib.use('svg')
# plt.switch_backend('agg')
matplotlib.use('svg')
import matplotlib.pyplot as plt


def train(args):
    ## load dataset
    adj_matrix, feature_matrix, labels, idx_train, _, idx_test = load(args.name)
    n_clusters = np.unique(labels).shape[0]
    # args.K = n_clusters
    n_nodes = feature_matrix.shape[0]
    fea_units = feature_matrix.shape[1]

    if args.name == 'cora' or args.name == 'citeseer':  # full-graph training
        n_samples = n_nodes
    else:
        n_samples = args.n_samples  # sub-graph training

    ## create model
    model = MVGRL(fea_units, args.hid_units, args.K, args.tau, args.gama)
    model = model.to(args.device) # The program may stop here unless you have enough GPU memory
    # print(model.state_dict().keys())
    view_learner = ViewLearner(fea_units, args.hid_units).to(args.device)
    # print(view_learner.state_dict().keys())

    ## setup optimizer
    view_optimizer = Adam(view_learner.parameters(), lr=args.view_lr)  #
    model_optimizer = Adam(model.parameters(), lr=args.lr)

    adj_matrix = torch.FloatTensor(adj_matrix)
    feature_matrix = torch.FloatTensor(feature_matrix)

    best_loss = 1e9
    best_epoch = 0
    cnt_wait = 0

    acc_reuslt = []
    acc_reuslt.append(0)
    nmi_result = []
    ari_result = []
    f1_result = []
    tag = 'pkl/' + str(args.name) + '/' + str(args.beta) + '_' + str(args.alpha) + '_' + str(args.K) + '_' + str(
        args.lambda_1)
    b_xent = nn.BCEWithLogitsLoss()
    if args.name in ['pubmed']:
        args.epoch = args.epoch*2
    for epoch in range(args.epoch):
        # --------------------------------------------min step under InfoMin principle with regularization terms-------------------------------------
        model.eval()
        model.zero_grad()
        view_learner.train()
        view_learner.zero_grad()

        if args.name == 'cora' or args.name == 'citeseer':  # full graph
            sub_adj_matrix = adj_matrix.to(args.device)
            sub_edge_index = torch.nonzero(sub_adj_matrix).t()
            sub_feature_matrix = feature_matrix.to(args.device)
        else:  # subgraph
            sampled_nodes = np.random.choice(n_nodes, size=n_samples, replace=False)
            sub_adj_matrix = adj_matrix[sampled_nodes, :][:, sampled_nodes].to(args.device)
            sub_edge_index = torch.nonzero(sub_adj_matrix).t()  # 
            sub_feature_matrix = feature_matrix[sampled_nodes, :].to(args.device)

        # -----feature masking augmenter
        edge_logits, fea_logits = view_learner(model.encoder, sub_feature_matrix, sub_adj_matrix,
                                               sub_edge_index)  # shape (M,1)
        aug_data_weight = torch.sigmoid(torch.mean(fea_logits, 0))
        aug_data_weight2 = aug_data_weight.expand_as(sub_feature_matrix).contiguous()
        aug_data = aug_data_weight2 * sub_feature_matrix

        # -----edge weight augmenter
        batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p
        aug_adj = torch.sparse.FloatTensor(sub_edge_index, batch_aug_edge_weight, size=sub_adj_matrix.shape)
        aug_adj = aug_adj.to_dense()
        aug_adj = aug_adj * sub_adj_matrix  #
        aug_adj = normalize_adj_torch(aug_adj, self_loop=True)  # A+I and normalize

        z_igae, aug_z_igae = model.embed(sub_feature_matrix, aug_data, sub_adj_matrix, aug_adj)

        view_loss = -args.beta * model.calc_loss(z_igae, aug_z_igae,
                                                 temperature=args.tau) + args.lambda_1 * batch_aug_edge_weight.mean() + args.lambda_2 * aug_data_weight.mean()

        view_loss.backward()
        # nn.utils.clip_grad_norm(view_learner.parameters(), 5, norm_type=2)
        view_optimizer.step()

        # --------------------------------------------max step under InfoMax principle with respective to node-global MI and node-cluster MI--
        view_learner.eval()
        view_learner.zero_grad()
        model.train()
        model.zero_grad()
        # -----feature masking augmenter
        edge_logits, fea_logits = view_learner(model.encoder, sub_feature_matrix, sub_adj_matrix, sub_edge_index)
        aug_data_weight = torch.sigmoid(torch.mean(fea_logits, 0))
        aug_data_weight2 = aug_data_weight.expand_as(sub_feature_matrix).contiguous()
        aug_data = aug_data_weight2 * sub_feature_matrix
        # -----edge weight augmenter
        batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p
        aug_adj = torch.sparse.FloatTensor(sub_edge_index, batch_aug_edge_weight, size=sub_adj_matrix.shape)
        aug_adj = aug_adj.to_dense()
        aug_adj = aug_adj * sub_adj_matrix  #
        aug_adj = normalize_adj_torch(aug_adj, self_loop=True)

        idx = np.random.permutation(n_samples)
        shuf_data = sub_feature_matrix[idx, :].to(args.device)
        shuf_aug_data = aug_data[idx, :].to(args.device)

        logits, logits2, z_igae, aug_z_igae = model(sub_feature_matrix, aug_data, shuf_data, shuf_aug_data,
                                                    sub_adj_matrix, aug_adj)
        lbl_1 = torch.ones(n_samples * 2)
        lbl_2 = torch.zeros(n_samples * 2)
        lbl = torch.cat((lbl_1, lbl_2)).to(args.device)

        model_loss = args.alpha * b_xent(logits, lbl) + (1 - args.alpha) * b_xent(logits2, lbl)

        model_loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), 5, norm_type=2)
        model_optimizer.step()

        # if epoch % 100 == 0:
        #     print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, model_loss.item()))
        #     z_igae, aug_z_igae= model.embed(sub_feature_matrix, aug_data, sub_adj_matrix, aug_adj)
        #
        #     embs = z_igae + aug_z_igae
        #     kmean = KMeans(n_clusters=int(n_clusters), n_init=20, random_state=seed)
        #     for z in [embs]:
        #         kmeans = kmean.fit(z.data.cpu().numpy())
        #         acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)

        if model_loss < best_loss:
            best_loss = model_loss
            best_epoch = epoch
            cnt_wait = 0
            model = model.eval()
            model.zero_grad()

            view_learner = view_learner.eval()
            view_learner.zero_grad()
            torch.save(view_learner.state_dict(), tag + '_view_learner.pkl')
            torch.save(model.state_dict(), tag + '_model.pkl')

        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    # --------------------------------------------eve step-------------------------------------
    print('loss{},Loading {}th epoch'.format(best_loss, best_epoch))
    view_learner.load_state_dict(torch.load(tag + '_view_learner.pkl'))
    model.load_state_dict(torch.load(tag + '_model.pkl'))

    # please put input data and model at cpu if you don't have enough GPU memory during test stage
    # print(model.device,view_learner.device)
    # # load dataset
    if args.name in ['pubmed','computers', 'photo','cs','physics']:
        model = model.cpu()
        view_learner = view_learner.cpu()
        args.device = 'cpu'

    model = model.eval()
    view_learner = view_learner.eval()
    # create new graph on cpu
    feature_matrix, adj_matrix = feature_matrix.to(args.device), adj_matrix.to(args.device)
    edge_index = torch.nonzero(adj_matrix).t()  # 找到A中非零元素的位置
    edge_logits, fea_logits = view_learner(model.encoder, feature_matrix.to(args.device), adj_matrix.to(args.device),
                                           edge_index)
    aug_data_weight = torch.sigmoid(torch.mean(fea_logits, 0))
    aug_data_weight2 = aug_data_weight.expand_as(feature_matrix).contiguous()
    aug_data = aug_data_weight2 * feature_matrix

    batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p
    aug_adj = torch.sparse.FloatTensor(edge_index, batch_aug_edge_weight, size=adj_matrix.shape)
    aug_adj = aug_adj.to_dense()
    aug_adj = aug_adj * adj_matrix
    aug_adj = normalize_adj_torch(aug_adj, self_loop=True)

    z_igae, aug_z_igae = model.embed(feature_matrix, aug_data, adj_matrix, aug_adj)

    embs = z_igae + aug_z_igae

    # ----clustering
    cacc, nmi, ari, f1 = 0, 0, 0, 0
    if 1 == 1 and args.name in ['cora', 'citeseer','pubmed']:
        kmean = KMeans(n_clusters=int(n_clusters), n_init=20, random_state=args.seed)
        kmeans = kmean.fit(embs.data.cpu().numpy())
        cacc, nmi, ari, f1 = eva(labels, kmeans.labels_, best_epoch)

    # ------classification
    if 1 == 1:
        # embs = embs2.data
        embs = embs.data
        train_embs = embs[idx_train].to(args.device)
        test_embs = embs[idx_test].to(args.device)
        labels = torch.LongTensor(labels)
        train_lbls = labels[idx_train].to(args.device)
        test_lbls = labels[idx_test].to(args.device)

        accs = []
        wd = 0.1 if args.name == 'citeseer' else 0.0
        xent = nn.CrossEntropyLoss()
        for _ in range(50):
            log = LogReg(args.hid_units, n_clusters).to(args.device)
            opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
            for _ in range(300):
                log.train()
                opt.zero_grad()
                logits = log(train_embs)
                loss = xent(logits, train_lbls)
                loss.backward()
                opt.step()

            log.eval()
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc * 100)

        accs = torch.stack(accs)
        print(accs.mean().item(), accs.std().item())

    return accs.mean().item(), accs.std().item(), cacc, nmi, ari, f1


def list2csv(my_list, file_name):
    import csv
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in my_list:
            writer.writerow(row)


if __name__ == '__main__':
    # path = '/home/guest3/workspace/tangqian/ACG-GCL_github/sensitivity/'
    import warnings
    print(torch.__version__)
    print(torch.cuda.is_available())
    warnings.filterwarnings("ignore")
    import argparse

    parser = argparse.ArgumentParser()
    # 'cora', 'citeseer', 'pubmed','computers', 'photo','cs','physics'

    parser.add_argument('--name', type=str, default='cora')
    parser.add_argument('--hid_units', type=int, default=512, help='embedding dimension')
    parser.add_argument('--n_samples', type=int, default=5000, help='number of samples')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--view_lr', type=float, default=1e-3, help='View Learning rate.')
    # min step
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--lambda_1', type=float, default=1)
    parser.add_argument('--lambda_2', type=float, default=1)
    # max step
    parser.add_argument('--alpha', type=int, default=0.5)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--gama', type=int, default=0.02)
    parser.add_argument('--tau', type=int, default=0.2)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--patience', type=int, default=75)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int,default=20230408)
    args = parser.parse_args()

    print(f"using cuda: {args.cuda}")
    device = torch.device(f"cuda:{args.cuda}" if args.cuda > -1 else "cpu")
    args.device = device
    # args.seed = random.randint(1, 10000)
    print(args.seed)
    setup_seed(args.seed)  #
    acc_mean, acc_std, cacc, nmi, ari, f1 = train(args)
    with open('log_{}.txt'.format(args.name), 'a') as f:
        f.write(str(args) + '\n')
        f.write(str(acc_mean) + '\t' + str(acc_std) + '\t' + str(cacc) + '\t' + str(nmi) + '\t' + str(
            ari) + '\t' + str(f1) + '\n')
