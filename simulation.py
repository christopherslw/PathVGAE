# simulation.py


import osmnx as ox
import networkx as nx
from model import PathVGAE
import random
import matplotlib.pyplot as plt
from utils import *
from torch_geometric.loader import DataLoader
import torch.nn as nn


train_load = ["SoCal1", "SoCal2", "SoCal3", "SoCal4", "SoCal5"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "graphs"
train_graphs = load_graphs(train_load, save_dir)
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)

model = PathVGAE(in_feats=5,hidden=32,num_layers=4,dropout=0.4,path_steps=4,vgae_hidden=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
mse_loss_fn = nn.MSELoss()

w_rank = 0.3
w_kl = 0.01
num_samples = 1000
rank_margin = 0.5
n_epochs = 100

for epoch in range(1, n_epochs+1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds, rank_scores, mu, logvar = model(batch.x, batch.edge_index)
        targets = batch.y.view_as(preds).float()
        mse = mse_loss_fn(preds, targets)
        ranking_edge_index = build_ranking(batch.y.view(-1),num_samples=num_samples,device=device)
        rank_loss = pairwise_rank_loss(rank_scores,ranking_edge_index,margin=rank_margin)
        kl_loss = kl_divergence(mu, logvar)
        loss = mse + w_rank * rank_loss + w_kl * kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    scheduler.step(avg_train_loss)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | train loss={avg_train_loss:.4f}")


######## Run simulation ###########

place = 'Santa Barbara, California, USA'
cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential"]'
G_sim = ox.graph_from_place(place, network_type='all_private', custom_filter=cf) # Sim1 graph
G_random = remove_edges_random(G_sim, 100) # Sim2 graph 
G_boundary = remove_edges_by_coordinates(G_sim, 34.43, 34.46, -119.715, -119.65) # Sim3 graph

sim_graphs = ['Sim1','Sim2','Sim3']
sim_dataset = load_graphs(sim_graphs,save_dir)
network_names=['Santa Barbara','Random','Boundary']
sim_dataloader = DataLoader(sim_dataset, batch_size=1, shuffle=False)
run_simulation(model, sim_dataloader, network_names, device)