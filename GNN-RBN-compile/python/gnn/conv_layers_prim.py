import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing

class MYACRConvPrim(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add', flow="source_to_target")
        self.ic = in_channels
        self.oc = out_channels
        self.ACR_bias = True

        self.A = torch.nn.Linear(in_channels, out_channels, bias=self.ACR_bias)
        self.R = torch.nn.Linear(in_channels, out_channels, bias=self.ACR_bias)
        self.C = torch.nn.Linear(in_channels, out_channels, bias=self.ACR_bias)

    def init(self):
        nn.init.uniform_(self.A.weight.data, -1, 1)
        nn.init.uniform_(self.R.weight.data, -1, 1)
        nn.init.uniform_(self.C.weight.data, -1, 1)
        if self.A.bias is not None:
            nn.init.uniform_(self.A.bias.data, -1, 1)
        if self.R.bias is not None:
            nn.init.uniform_(self.R.bias.data, -1, 1)
        if self.C.bias is not None:
            nn.init.uniform_(self.C.bias.data, -1, 1)   
    
    def forward(self, x, edge_index):
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        r = global_add_pool(x, batch)
        r = r[batch]
        out = self.propagate(edge_index, h=x, readout=r)
        return out

    def message(self, h_j):
        return h_j

    def update(self, aggr, h, readout):    
        updated = self.A(aggr) + self.C(h) + self.R(readout)# + self.b
        return torch.sigmoid(updated)