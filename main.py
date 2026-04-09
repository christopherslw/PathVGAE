# main.py
import torch
from model import PathVGAE
from utils import *
from scipy.stats import kendalltau
from torch_geometric.loader import DataLoader
import torch.nn as nn


train_load = ["SoCal1", "SoCal2", "SoCal3", "SoCal4", "SoCal5"]
test_load = ["Santa_Clara_County_California_USA",
             "District_of_Columbia_USA",
             "Yellowstone_County_Montana_USA",
             "Hennepin_County_Minnesota_USA",
             "Maricopa_County_Arizona_USA"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "graphs"
train_graphs = load_graphs(train_load, save_dir)
test_graphs = load_graphs(test_load, save_dir)
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)

model = PathVGAE(in_feats=5,hidden=32,num_layers=4,dropout=0.4,path_steps=4,vgae_hidden=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
mse_loss_fn = nn.MSELoss()

w_rank = 0.5
w_kl = 0.01
num_samples = 1000
rank_margin = 0.5
n_epochs = 200

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
        loss = mse + w_rank*rank_loss + w_kl*kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    scheduler.step(avg_train_loss)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | train loss={avg_train_loss:.4f}")

print("Testing on different counties")
model.eval()
for name, graph in zip(test_load, test_graphs):
    test_data = graph.to(device)
    with torch.no_grad():
        preds = model.predict(test_data.x, test_data.edge_index).cpu().numpy().ravel()
        targets = test_data.y.cpu().numpy().ravel()
    test_tau = kendalltau(preds, targets).correlation
    print(f"{name}: Kendall Tau={test_tau:.4f}")