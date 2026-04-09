# utils.py

import os
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from scipy.stats import kendalltau
import random
import matplotlib.pyplot as plt


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


def load_graphs(names, save_dir):
    graphs = []
    for name in names:
        path = os.path.join(save_dir, f"{name}.pt")
        if os.path.exists(path):
            graphs.append(torch.load(path, weights_only=False))
        else:
            print(f"{path} not found!!!")
    return graphs


def build_ranking(y, num_samples=200, device="cpu"):
    y = y.view(-1)
    N = y.numel()
    num_samples = min(num_samples, N)

    idx = torch.randperm(N, device=device)[:num_samples]
    scores = y[idx]

    i_idx, j_idx = torch.where(scores[:, None] > scores[None, :])

    if i_idx.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    src = idx[i_idx]
    dst = idx[j_idx]
    return torch.stack([src, dst], dim=0)


def pairwise_rank_loss(rank_scores, ranking_edge_index, margin=0.5):
    """
    hinge ranking loss
    """
    if ranking_edge_index.numel() == 0:
        return rank_scores.new_tensor(0.0)
    src, dst = ranking_edge_index
    return F.relu(margin - (rank_scores[src] - rank_scores[dst])).mean()

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())



def vgae_loss(mu, logvar, ranking_edge_index, margin=0.5):
    pos_src, pos_dst = ranking_edge_index
    pos_i = mu[pos_src].norm(dim=-1)
    pos_j = mu[pos_dst].norm(dim=-1)
    rank_loss = F.relu(margin - (pos_i - pos_j)).mean()
    kl_loss   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rank_loss, kl_loss


def remove_edges_random(G, num_edges_to_remove=100):
    """
    simulate random road closures via random edge removal on a graph.
    """
    G_sim = G.copy()
    edges = list(G_sim.edges(keys=True))

    num_edges_to_remove = min(num_edges_to_remove, len(edges))
    random_edges = random.sample(edges, num_edges_to_remove)

    for u, v, key in random_edges:
        if G_sim.has_edge(u, v, key):
            G_sim.remove_edge(u, v, key)

    gdf_edges_updated = ox.graph_to_gdfs(G_sim, nodes=False, edges=True)
    gdf_edges_updated = gdf_edges_updated.to_crs(epsg=4326)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_edges_updated.plot(ax=ax, linewidth=1, edgecolor='grey', alpha=0.7)
    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)  # Get original edges for reference
    gdf_edges_removed = gdf_edges.loc[gdf_edges.index.isin(random_edges)]
    gdf_edges_removed.plot(ax=ax, linewidth=2, edgecolor='darkred', alpha=1.0)
    plt.show()

    return G_sim


def remove_edges_by_coordinates(G, region_lat_min, region_lat_max, region_lon_min, region_lon_max):
    """
    simulate closures of specific areas by removing edges whose centroids
    are greater than the latitude and longitude bounds.
    """
    G_sim = G.copy()
    gdf_edges = ox.graph_to_gdfs(G_sim, nodes=False, edges=True)
    gdf_edges = gdf_edges.to_crs(epsg=4326)

    edges_in_region = gdf_edges[
        (gdf_edges['geometry'].centroid.y >= region_lat_min) &
        (gdf_edges['geometry'].centroid.y <= region_lat_max) &
        (gdf_edges['geometry'].centroid.x >= region_lon_min) &
        (gdf_edges['geometry'].centroid.x <= region_lon_max)
    ]

    for u, v, key in edges_in_region.index:
        if G_sim.has_edge(u, v, key):
            G_sim.remove_edge(u, v, key)

    gdf_edges_updated = ox.graph_to_gdfs(G_sim, nodes=False, edges=True)
    gdf_edges_updated = gdf_edges_updated.to_crs(epsg=4326)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_edges_updated.plot(ax=ax, linewidth=1, edgecolor='grey', alpha=0.7)
    gdf_edges_removed = gdf_edges.loc[gdf_edges.index.isin(edges_in_region.index)]
    gdf_edges_removed.plot(ax=ax, linewidth=2, edgecolor='darkred', alpha=1.0)
    plt.show()

    return G_sim


def run_simulation(model, sim_dataloader, network_names, device):
    print("Running simulation.")
    for graph, network_name in zip(sim_dataloader, network_names):
        sim_data = graph.to(device)
        model.eval()
        with torch.no_grad():
            preds = model.predict(sim_data.x, sim_data.edge_index).cpu().numpy().ravel()
            targets = sim_data.y.cpu().numpy().ravel()
        sim_tau = kendalltau(preds, targets).correlation
        print(f"{network_name} KT Score: {sim_tau}")


