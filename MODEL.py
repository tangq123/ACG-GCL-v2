import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU


class ViewLearner(torch.nn.Module):
    def __init__(self, msk_fea_dim, hid_units, requires_grad=True):
        super(ViewLearner, self).__init__()

        self.mlp_edge_model = Linear(hid_units * 2, 1, requires_grad)
        self.mlp_fea_masking_model = Linear(hid_units, msk_fea_dim, requires_grad)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, encoder, x, adj, edge_index):
        node_emb = encoder(x, adj)
        src, dst = edge_index[0], edge_index[1]
        edge_emb = torch.cat([node_emb[src], node_emb[dst]], 1)
        edge_logits = self.mlp_edge_model(edge_emb)
        fea_logits = self.mlp_fea_masking_model(node_emb)
        return edge_logits, fea_logits


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, c1, c2, h1, h2, h3, h4):
        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()

        # positive
        sc_1 = self.fn(h2, c_x1).squeeze(1)
        sc_2 = self.fn(h1, c_x2).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x1).squeeze(1)
        sc_4 = self.fn(h3, c_x2).squeeze(1)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4))

        return logits


class Discriminator_cluster(nn.Module):
    def __init__(self, n_h):
        super(Discriminator_cluster, self).__init__()

        self.n_h = n_h

    def forward(self, z_1, z_2, h_1, h_2, h_3, h_4):
        sc_1 = torch.bmm(h_2.view(-1, 1, self.n_h), z_1.view(-1, self.n_h, 1)).squeeze()
        sc_2 = torch.bmm(h_1.view(-1, 1, self.n_h), z_2.view(-1, self.n_h, 1)).squeeze()
        sc_3 = torch.bmm(h_4.view(-1, 1, self.n_h), z_1.view(-1, self.n_h, 1)).squeeze()
        sc_4 = torch.bmm(h_3.view(-1, 1, self.n_h), z_2.view(-1, self.n_h, 1)).squeeze()

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4))

        return logits


class Clusterator(nn.Module):

    def __init__(self, n_h, K, gama):
        super(Clusterator, self).__init__()

        self.n_h = n_h
        self.K = K
        self.init = torch.rand(self.K, n_h)  #
        self.gama = gama

    def forward(self, embeds, num_iter=11):
        # mu = torch.rand(self.K, self.n_h).to(self.device) 
        mu = self.init.to(embeds.device)
        data = embeds
        eps = 1e-8
        data_abs = data.norm(dim=1).unsqueeze(1)
        data_norm = data / torch.max(data_abs, eps * torch.ones_like(data_abs))

        for _ in range(num_iter):
            mu_abs = mu.norm(dim=1).unsqueeze(1)
            mu_norm = mu / torch.max(mu_abs, eps * torch.ones_like(mu_abs))
            dist = torch.mm(data_norm, mu_norm.transpose(0, 1))
            r = F.softmax(dist / self.gama, dim=1)
            cluster_r = r.sum(dim=0)
            cluster_mean = torch.mm(r.t(), data)
            new_mu = torch.mm(torch.diag(1 / cluster_r), cluster_mean)
            mu = new_mu

        return mu, r


class MVGRL(nn.Module):
    def __init__(self, n_in, n_h, num_clusters, tau, gama):
        super(MVGRL, self).__init__()
        self.encoder = GCN(n_in, n_h)

        self.read = Readout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.disc_c = Discriminator_cluster(n_h)

        self.tau = tau

        self.cluster1 = Clusterator(n_h, num_clusters, gama)
        self.cluster2 = Clusterator(n_h, num_clusters, gama)

    def forward(self, seq1, seq2, seq3, seq4, adj, diff):

        h_1 = self.encoder(seq1, adj)

        c_1 = self.read(h_1)
        c_1 = self.sigm(c_1)

        h_2 = self.encoder(seq2, diff)
        c_2 = self.read(h_2)
        c_2 = self.sigm(c_2)

        # cluster summary from view1
        mu1, r1 = self.cluster1(h_1)  #
        z_1 = r1 @ mu1
        z_1 = self.sigm(z_1)  #
        # cluster summary from view2
        mu2, r2 = self.cluster2(h_2)  #
        z_2 = r2 @ mu2
        z_2 = self.sigm(z_2)

        # negative embeddings
        h_3 = self.encoder(seq3, adj)
        h_4 = self.encoder(seq4, diff)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)
        ret2 = self.disc_c(z_1, z_2, h_1, h_2, h_3, h_4)

        return ret, ret2, h_1, h_2

    def embed(self, seq1, seq2, adj, diff):
        h_1 = self.encoder(seq1, adj)

        h_2 = self.encoder(seq2, diff)

        return h_1, h_2

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True, eps=1e-8):

        x_abs = x.norm(dim=1).unsqueeze(1)
        x_norm = x / torch.max(x_abs, eps * torch.ones_like(x_abs))
        x_aug_abs = x_aug.norm(dim=1).unsqueeze(1)
        x_aug_norm = x_aug / torch.max(x_aug_abs, eps * torch.ones_like(x_aug_abs))
        sim_matrix = torch.einsum('bc,cd->bd', x_norm, x_aug_norm.t())
        sim_matrix = torch.exp(sim_matrix / temperature)

        batch_size = x.shape[0]
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
        else:
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()
        return loss


class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class ACG_GCL(nn.Module):
    def __init__(self, n_in, n_h, num_clusters, tau, gama, requires_grad):
        super(ACG_GCL, self).__init__()
        self.encoder = GCN(n_in, n_h)
        self.view_learner = ViewLearner(n_in, n_h, requires_grad)

        self.read = Readout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.disc_c = Discriminator_cluster(n_h)

        self.tau = tau

        self.cluster1 = Clusterator(n_h, num_clusters, gama)
        self.cluster2 = Clusterator(n_h, num_clusters, gama)

    def forward(self, seq1, seq2, seq3, seq4, adj, diff):
        h_1 = self.encoder(seq1, adj)

        c_1 = self.read(h_1)
        c_1 = self.sigm(c_1)

        h_2 = self.encoder(seq2, diff)
        c_2 = self.read(h_2)
        c_2 = self.sigm(c_2)

        # cluster summary from view1
        mu1, r1 = self.cluster1(h_1)  #
        z_1 = r1 @ mu1
        z_1 = self.sigm(z_1)  #
        # cluster summary from view2
        mu2, r2 = self.cluster2(h_2)  #
        z_2 = r2 @ mu2
        z_2 = self.sigm(z_2)

        # negative embeddings
        h_3 = self.encoder(seq3, adj)
        h_4 = self.encoder(seq4, diff)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)
        ret2 = self.disc_c(z_1, z_2, h_1, h_2, h_3, h_4)

        return ret, ret2, h_1, h_2

    def embed(self, seq1, seq2, adj, diff):
        h_1 = self.encoder(seq1, adj)
        h_2 = self.encoder(seq2, diff)
        return h_1, h_2
