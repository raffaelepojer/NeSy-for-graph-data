import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from .conv_layers import MYACRConv

# base code taken from https://github.com/juanpablos/GNN-logic

class MYACRGnnNode(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, final_read="add", num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.layers = torch.nn.ModuleList()

        if final_read == "add":
            self.final_readout = global_add_pool
        elif final_read == "mean":
            self.final_readout = global_mean_pool
        else:
            print("Final readout {0} not implemented!".format(final_read))

        if type(hidden_dim) == list:
            for l in range(num_layers):
                if l == 0:
                    self.layers.append(MYACRConv(input_dim, hidden_dim[l]))
                else:
                    self.layers.append(MYACRConv(hidden_dim[l-1], hidden_dim[l]))
            self.linear = torch.nn.Linear(hidden_dim[-1], num_classes)            
        else:
            print("Hidden dim {0} not implemented! Must be a list".format(hidden_dim))
    

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer.forward(x, edge_index, batch)    
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
    
    def init_model(self):
        nn.init.uniform_(self.linear.weight, -1, 1)
        nn.init.uniform_(self.linear.bias, -1, 1)
        for layer in self.layers:
            layer.init()
    