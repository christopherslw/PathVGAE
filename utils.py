# utils.py

import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import kendalltau


def get_data_from_place(place, cf, mode='betweenness'):
    G = ox.graph_from_place(place, network_type='all_private', custom_filter=cf)
    dual_G = nx.line_graph(G)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_nodes = gdf_nodes.reset_index()

    if mode == 'betweenness':
        centrality = nx.betweenness_centrality(dual_G, normalized=False)
        centrality_column = 'betweenness_centrality'
    elif mode == 'closeness':
        centrality = nx.closeness_centrality(dual_G)
        centrality_column = 'closeness_centrality'
    else:
        raise ValueError(f"Unsupported mode: '{mode}'. Choose from: 'betweenness', 'closeness'")

    centrality_df = pd.DataFrame.from_dict(centrality, orient='index', columns=[centrality_column])
    centrality_df = centrality_df.reset_index()
    centrality_df.columns = ['osmid', centrality_column]  # 'osmid' is the unique node identifier in gdf_nodes

    gdf = pd.merge(gdf_edges, centrality_df, on='osmid', how='left')

    adj_matrix = nx.to_scipy_sparse_array(dual_G)
    adj = torch.tensor(adj_matrix.toarray(), dtype=torch.float32)
    features = torch.tensor(gdf['length'].values, dtype=torch.float32)
    targets = torch.tensor(gdf[centrality_column].values, dtype=torch.float32)

    return dual_G, features, adj, targets


def get_data_from_bbox(north, south, east, west, cf, mode='betweenness'):
    G = ox.graph_from_bbox(north, south, east, west, network_type='all_private', custom_filter=cf)
    dual_G = nx.line_graph(G)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_nodes = gdf_nodes.reset_index()

    if mode == 'betweenness':
        centrality = nx.betweenness_centrality(dual_G, normalized=False)
        centrality_column = 'betweenness_centrality'
    elif mode == 'closeness':
        centrality = nx.closeness_centrality(dual_G)
        centrality_column = 'closeness_centrality'
    else:
        raise ValueError(f"Unsupported mode: '{mode}'. Choose from: 'betweenness', 'closeness'")

    centrality_df = pd.DataFrame.from_dict(centrality, orient='index', columns=[centrality_column])
    centrality_df = centrality_df.reset_index()
    centrality_df.columns = ['osmid', centrality_column]  # 'osmid' is the unique node identifier in gdf_nodes

    gdf = pd.merge(gdf_edges, centrality_df, on='osmid', how='left')

    adj_matrix = nx.to_scipy_sparse_array(dual_G)
    adj = torch.tensor(adj_matrix.toarray(), dtype=torch.float32)
    features = torch.tensor(gdf['length'].values, dtype=torch.float32)
    targets = torch.tensor(gdf[centrality_column].values, dtype=torch.float32)

    return dual_G, features, adj, targets


class GraphDataset(Dataset):
    def __init__(self, graphs, node_features, targets, model_size):
        self.graphs = graphs
        self.node_features = node_features
        self.targets = targets
        self.model_size = model_size

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        node_feature = self.node_features[idx]
        target = self.targets[idx]

        current_size = len(graph)
        adj_matrix = nx.to_numpy_array(graph)
        padded_matrix = np.zeros((self.model_size, self.model_size), dtype=adj_matrix.dtype)
        padded_matrix[:current_size, :current_size] = adj_matrix

        if len(node_feature.shape) == 1:
            node_feature = node_feature.reshape(-1, 1)

        feature_size = min(current_size, node_feature.shape[0])

        padded_features = np.zeros((self.model_size, node_feature.shape[1]), dtype=node_feature.numpy().dtype)
        padded_features[:feature_size, :] = node_feature.numpy()[:feature_size, :]
        padded_target = np.zeros((self.model_size,), dtype=target.numpy().dtype)
        padded_target[:feature_size] = target.numpy()[:feature_size]

        return {
            'adj': torch.tensor(padded_matrix, dtype=torch.float32),
            'features': torch.tensor(padded_features, dtype=torch.float32),
            'target': torch.tensor(padded_target, dtype=torch.float32)
        }


def calculate_kendall_tau(pred, target):
    '''
    Calculates Kendall Tau ranking correlation for the predicted
    centrality rankings vs the true centrality rankigs
    '''
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    tau, _ = kendalltau(pred, target)
    return tau


def rank_loss(y_out, true_val, current_size):
    """Calculates ranking based loss function."""
    # use the number of valid nodes instead of fixed model_size
    y_out = y_out[:current_size].reshape((current_size))
    true_val = true_val[:current_size].reshape((current_size))

    _, order_y_true = torch.sort(-true_val)

    sample_num = current_size * 20

    ind_1 = torch.randint(0, current_size, (sample_num,)).long()
    ind_2 = torch.randint(0, current_size, (sample_num,)).long()

    rank_measure = torch.sign(-1 * (ind_1 - ind_2)).float()

    input_arr1 = y_out[order_y_true[ind_1]]
    input_arr2 = y_out[order_y_true[ind_2]]

    loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1, input_arr2, rank_measure)

    return loss_rank


def kt_loss(y_out, true_val, alpha=10.0):
    """
    Differentiable approximation of Kendall's Tau as a loss function.
    """

    y_out = y_out.view(-1)
    true_val = true_val.view(-1)

    n = y_out.shape[0]

    if n < 2:
        print("Warning: y_out and true_val contain fewer than 2 elements. Returning a loss of 0.")
        return torch.tensor(0.0)

    ind_i, ind_j = torch.triu_indices(n, n, offset=1)

    y_out_diff = y_out[ind_i] - y_out[ind_j]
    true_val_diff = true_val[ind_i] - true_val[ind_j]

    y_out_sign_approx = torch.tanh(alpha * y_out_diff)
    true_val_sign_approx = torch.tanh(alpha * true_val_diff)

    kendall_tau_approx = torch.mean(y_out_sign_approx * true_val_sign_approx)
    loss = 1 - kendall_tau_approx

    return loss
