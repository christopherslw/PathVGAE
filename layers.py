# layers.py

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
from utils import *

class GNNEmbedLayer(nn.Module):
    """
    Embedding layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GNNEmbedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj):
        support = self.weight
        output = torch.spmm(adj.squeeze(0), support)  # remove batch dimension
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias

        return torch.sparse.mm(adj.squeeze(0), x)


class PathConvLayer(nn.Module):
    def __init__(self, in_features, out_features, aggregator='mean', use_bias=True, num_samples=50, path_length=3):
        super(PathConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        self.num_samples = num_samples
        self.path_length = path_length
        self.weight = nn.Parameter(torch.FloatTensor(2 * in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def sample_paths(self, adj_matrix, num_paths, path_length):
        """
        sample a fixed number of neighbors for each node from a dense adjacency matrix
        """
        if adj_matrix.dim() == 3 and adj_matrix.size(0) == 1:
            adj_matrix = adj_matrix.squeeze(0)

        num_nodes = adj_matrix.shape[0]
        sampled_paths = []

        for _ in range(num_paths):
            u = random.randint(0, num_nodes - 1)
            path = []

            for _ in range(path_length - 1):
                neighbors = torch.nonzero(adj_matrix[u]).squeeze()
                if neighbors.dim() == 0:
                    neighbors = neighbors.unsqueeze(0)

                neighbors = neighbors.tolist()
                if len(neighbors) > 0:
                    v = random.choice(neighbors)
                    path.append(v)
                    u = v
                else:
                    break

            sampled_paths.extend(path)
            return [np.array(sampled_paths)]

    def aggregate(self, x, neighbor_list):
        """
        Aggregate features from sampled neighbors
        """
        out = torch.zeros_like(x)
        for i, neighbors in enumerate(neighbor_list):
            if len(neighbors) == 0:
                out[i] = x[i]
                continue
            if self.aggregator == 'mean':
                neighbor_feats = torch.mean(x[neighbors], dim=0)
            elif self.aggregator == 'sum':
                neighbor_feats = torch.sum(x[neighbors], dim=0)
            elif self.aggregator == 'max':
                neighbor_feats, _ = torch.max(x[neighbors], dim=0)
            else:
                raise NotImplementedError("Aggregator type not supported.")
            out[i] = neighbor_feats
        return out

    def forward(self, x, adj):
        """
        Forward pass for the GraphSAGE layer
        """
        neighbor_list = self.sample_paths(adj, self.num_samples, self.path_length)
        agg_feats = self.aggregate(x, neighbor_list)

        x = torch.cat([x, agg_feats], dim=1)
        x = x @ self.weight

        if self.bias is not None:
            x += self.bias

        return F.relu(x)
