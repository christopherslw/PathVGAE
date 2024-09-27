# simulation.py

from utils import *
import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt
from main import train_model

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



def run_simulation(model, mode):
    if mode == 'betweenness':
        sim_dataset = torch.load('Data/sim_data_betweenness.pth')
    elif mode == 'closeness':
        sim_dataset = torch.load('Data/sim_data_closeness.pth')
    else:
        raise ValueError(f"Unsupported centrality type: '{mode}'")

    sim_dataloader = DataLoader(sim_dataset, batch_size=1, shuffle=False)
    network_names = ['Santa Barbara', 'Random', 'Boundary']
    model.eval()
    with torch.no_grad():
        for batch, network_name in zip(sim_dataloader, network_names):
            adj_pad_sim = batch['adj']
            adj_pad_t_sim = adj_pad_sim.transpose(1, 2)
            target_sim = batch['target']
            current_size = target_sim.shape[1]

            y_sim_pred = model(adj_pad_sim, adj_pad_t_sim)
            sim_kendall_tau = calculate_kendall_tau(y_sim_pred[:current_size], target_sim[:current_size])
            print(f"{network_name} KT Score: {sim_kendall_tau}")



######## Run simulation ###########

place = 'Santa Barbara, California, USA'
cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential"]'
G_sim = ox.graph_from_place(place, network_type='all_private', custom_filter=cf)

G_random = remove_edges_random(G_sim, 100) # example
G_area = remove_edges_by_coordinates(G_sim, 34.43, 34.46, -119.715, -119.65)

model, mode = train_model(mode='betweenness', print_kt=False)
run_simulation(model, mode)
