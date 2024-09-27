# models.py

from layers import GNNEmbedLayer, PathConvLayer
import torch.nn as nn
import torch
import torch.nn.functional as F


class PathVGAE(nn.Module):
    def __init__(self, ninput, nhid, dropout):
        super(PathVGAE, self).__init__()
        # encoder layers
        self.gc1 = GNNEmbedLayer(ninput, nhid) # embedding
        self.gc2 = PathConvLayer(nhid, nhid)
        self.gc3 = PathConvLayer(nhid, nhid)
        self.gc4 = PathConvLayer(nhid, nhid)

        # decoder MLP layers
        self.dropout = dropout
        self.linear1 = nn.Linear(nhid, 2 * nhid)
        self.linear2 = nn.Linear(2 * nhid, 2 * nhid)
        self.linear3 = nn.Linear(2 * nhid, 1)


    def reparametrize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            return mu


    def encode(self, adj1, adj2):
        mu = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
        logvar = F.normalize(F.relu(self.gc1(adj2)), p=2, dim=1)

        mu = F.normalize(F.relu(self.gc2(mu, adj1)), p=2, dim=1)
        logvar = F.normalize(F.relu(self.gc2(logvar, adj2)), p=2, dim=1)

        mu = F.normalize(F.relu(self.gc3(mu, adj1)), p=2, dim=1)
        logvar = F.normalize(F.relu(self.gc3(logvar, adj2)), p=2, dim=1)

        mu = F.relu(self.gc4(mu, adj1))
        logvar = F.relu(self.gc4(logvar, adj2))

        return mu, logvar


    def decode(self, mu, logvar):
        s1 = F.relu(self.linear1(mu))
        s1 = F.dropout(s1, self.dropout)
        s1 = F.relu(self.linear2(s1))
        s1 = F.dropout(s1, self.dropout)
        s1 = self.linear3(s1)

        s2 = F.relu(self.linear1(logvar))
        s2 = F.dropout(s2, self.dropout)
        s2 = F.relu(self.linear2(s2))
        s2 = F.dropout(s2, self.dropout)
        s2 = self.linear3(s2)

        pred = torch.mul(s1, s2)

        return pred


    def forward(self, adj1, adj2):
        mu, logvar = self.encode(adj1, adj2)
        output = self.decode(mu, logvar)
        return output
