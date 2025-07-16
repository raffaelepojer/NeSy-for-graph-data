import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, GCN2Conv, GATConv
import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class HeteroGraph(nn.Module):
    def __init__(self, in_sub, in_hru_agr, in_hru_urb, hidden_dims, out_dims, num_layers=3, dropout=0.5, sage_aggr='sum', batch_norm=False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() 
        self.res_proj = nn.ModuleList()  
        self.heteroAggr = 'sum'
        self.batch_norm = batch_norm

        self.out_dim_sub = hidden_dims * 1
        if self.heteroAggr == 'cat':
            self.out_dim_sub = hidden_dims * 3

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                conv_dict = {
                    ('sub', 'downstream', 'sub'): SAGEConv(in_sub, hidden_dims, aggr=sage_aggr),
                    ('agr', 'downstream_agr', 'sub'): SAGEConv((in_hru_agr, in_sub), hidden_dims, aggr=sage_aggr),
                    # ('sub', 'upstream_agr', 'agr'): SAGEConv((in_sub, in_hru_agr), hidden_dims),
                    ('urb', 'downstream_urb', 'sub'): SAGEConv((in_hru_urb, in_sub), hidden_dims, aggr=sage_aggr),
                    # ('sub', 'upstream_urb', 'urb'): SAGEConv((in_sub, in_hru_urb), hidden_dims)
                }
            else:
                conv_dict = {
                    ('sub', 'downstream', 'sub'): SAGEConv(self.out_dim_sub, hidden_dims),
                    ('agr', 'downstream_agr', 'sub'): SAGEConv((in_hru_agr, self.out_dim_sub), hidden_dims, aggr=sage_aggr),
                    # ('sub', 'upstream_agr', 'agr'): SAGEConv((self.out_dim_sub, in_hru_agr), hidden_dims),
                    ('urb', 'downstream_urb', 'sub'): SAGEConv((in_hru_urb, self.out_dim_sub), hidden_dims, aggr=sage_aggr),
                    # ('sub', 'upstream_urb', 'urb'): SAGEConv((self.out_dim_sub, in_hru_urb), hidden_dims)
                }

            self.layers.append(HeteroConv(conv_dict, aggr=self.heteroAggr))
            if self.batch_norm:
                self.norms.append(nn.BatchNorm1d(self.out_dim_sub))
            else:
                self.norms.append(nn.Identity())

        self.final_lin = nn.Linear(self.out_dim_sub, out_dims)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        skip_hru_agr = x_dict['agr']
        skip_hru_urb = x_dict['urb']
        for i, layer in enumerate(self.layers):
            x_dict = layer(x_dict, edge_index_dict)

            x_dict['sub'] = F.relu(self.norms[i](x_dict['sub']))
            x_dict['sub'] = self.dropout(x_dict['sub'])
            
            # Reapply the original features for non-'sub' nodes (as skip conn)
            x_dict['agr'] = skip_hru_agr
            x_dict['urb'] = skip_hru_urb

        out = self.final_lin(x_dict['sub'])
        return F.softmax(out, dim=1)