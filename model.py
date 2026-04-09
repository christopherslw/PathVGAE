# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from utils import *


class PathVGAE(nn.Module):

    def __init__(self,in_feats=5,hidden=32,num_layers=4,dropout=0.4,path_steps=4,vgae_hidden=32):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.path_steps = path_steps
        self.vgae_hidden = vgae_hidden
        layer_out = hidden
        self.input_proj = nn.Linear(in_feats, hidden)
        self.fwd_convs = nn.ModuleList()
        self.fwd_bns = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        conv_in = in_feats
        for _ in range(num_layers):
            self.fwd_convs.append(SAGEConv(conv_in, hidden))
            self.fwd_bns.append(nn.BatchNorm1d(hidden))
            if conv_in != layer_out:
                self.res_projs.append(nn.Linear(conv_in,layer_out))
            else:
                self.res_projs.append(nn.Identity())                        
            conv_in = layer_out

        stage1_dim = hidden*num_layers + hidden
        vgae_in = stage1_dim + hidden
        self.vgae_fwd_conv = SAGEConv(vgae_in, vgae_hidden)
        self.vgae_fwd_bn = nn.BatchNorm1d(vgae_hidden)
        self.mu_proj = nn.Linear(vgae_hidden, vgae_hidden)
        self.logvar_proj = nn.Linear(vgae_hidden, vgae_hidden)

        self.rank_head = nn.Sequential(nn.Linear(vgae_hidden,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        reg_in = stage1_dim + vgae_hidden + 1
        self.regression_head = nn.Sequential(nn.Linear(reg_in, hidden),nn.ReLU(),nn.Dropout(dropout),
                                             nn.Linear(hidden, hidden//2),nn.ReLU(),nn.Linear(hidden//2,1))


    def path_sample_agg(self, h, edge_index, num_nodes, steps=4):
        if edge_index.numel() == 0:
            return h
        row, col = edge_index
        deg = degree(row, num_nodes=num_nodes).long()
        perm = row.argsort()
        row_sorted = row[perm]
        col_sorted = col[perm]
        max_deg = int(deg.max().item())
        if max_deg == 0:
            return h
        
        neighbors = h.new_full((num_nodes,max_deg),-1,dtype=torch.long)
        counts = deg.cumsum(0)
        offsets = torch.cat([deg.new_zeros(1),counts[:-1]])
        pos = torch.arange(row_sorted.numel(),device=h.device) - offsets[row_sorted]
        neighbors[row_sorted,pos] = col_sorted
        current = torch.arange(num_nodes,device=h.device,dtype=torch.long).unsqueeze(1).expand(num_nodes,self.path_steps)
        path_sum = h.unsqueeze(1).expand(num_nodes,self.path_steps,h.size(-1)).clone()
        for _ in range(steps):
            deg_cur = deg[current]
            rand_idx = (torch.rand_like(deg_cur.float())*deg_cur.clamp(min=1).float()).long()
            next_nodes = neighbors[current,rand_idx]
            next_nodes = torch.where(deg_cur>0,next_nodes,current)
            path_sum = path_sum+h[next_nodes]
            current = next_nodes

        out = path_sum.mean(dim=1) / (steps + 1)
        return out

    def path_agg(self,h,edge_index,num_nodes,steps=4): # diffusion 
        if edge_index.numel() == 0:
            return h
        row, col = edge_index
        deg = degree(row, num_nodes=num_nodes).clamp(min=1)
        norm = (1.0/deg)[row]
        out = h
        for _ in range(steps):
            agg = out.new_zeros(out.shape)
            agg.index_add_(0,col,out[row]*norm.unsqueeze(-1))
            out = agg
        return out

    def encode(self,x,edge_index):
        N = x.size(0)
        inp_skip = self.input_proj(x)
        h = x
        layer_outputs = []

        for i in range(self.num_layers):
            h_res = self.res_projs[i](h)
            h_fwd = F.relu(self.fwd_bns[i](self.fwd_convs[i](h, edge_index)))
            h = h_fwd + h_res
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outputs.append(h)

        stage1 = torch.cat(layer_outputs+[inp_skip], dim=1)
        #h_agg = self.path_sample_agg(h, edge_index, N, steps=self.path_steps)
        h_agg = self.path_agg(h, edge_index, N, steps=self.path_steps)
        vgae_in = torch.cat([stage1, h_agg], dim=1)
        v_fwd = F.relu(self.vgae_fwd_bn(self.vgae_fwd_conv(vgae_in, edge_index)))
        mu = self.mu_proj(v_fwd)
        logvar = self.logvar_proj(v_fwd)

        if self.training:
            z = mu + torch.randn_like(mu)*torch.exp(0.5 * logvar)
        else:
            z = mu

        rank_score = self.rank_head(z).squeeze(-1)  # [N]
        return stage1, rank_score, mu, logvar, z

    def forward(self, x, edge_index):
        stage1, rank_score, mu, logvar, z = self.encode(x, edge_index)
        reg_in = torch.cat([stage1, z, rank_score.unsqueeze(-1)], dim=1)
        preds = self.regression_head(reg_in)
        return preds, rank_score, mu, logvar

    def predict(self, x, edge_index):
        preds, _, _, _ = self.forward(x, edge_index)
        return preds

