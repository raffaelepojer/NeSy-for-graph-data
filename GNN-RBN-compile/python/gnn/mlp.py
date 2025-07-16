import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    # MLP with linear output
    def __init__(self,
                 num_layers:int,
                 input_dim:int,
                 hidden_dim:int,
                 output_dim:int,
                 mlp_dp: float = 0.0,
                 batch_norm: bool = False,
                 activation: str = 'sigmoid'
                 ):
        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.batch_norms = batch_norm
        self.mlp_dp = mlp_dp
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'relu':
            self.activation = torch.relu

        if num_layers < 1:
            self.linear = nn.Identity()
        else:
            # Multi-layer model
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            if self.num_layers > 1:
                self.linears.append(nn.Linear(input_dim, hidden_dim))
                for _ in range(num_layers - 2):
                    self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                self.linears.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.linears.append(nn.Linear(input_dim, output_dim))

            if batch_norm:
                for layer in range(num_layers - 1):
                    self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        h = x
        for layer in range(self.num_layers - 1):
            if self.batch_norms:
                h = self.activation(self.batch_norms[layer](self.linears[layer](h)))
            else:
                h = F.dropout(h, p=self.mlp_dp, training=self.training)
                h = self.activation(self.linears[layer](h))
        return self.linears[self.num_layers - 1](h)