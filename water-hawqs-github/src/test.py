from model import HeteroGraph, EarlyStopper
import torch
from sklearn.metrics import f1_score
import pickle
import random
import numpy as np

with open('../data/train_graphs_all.pkl', 'rb') as f:
    train_graphs = pickle.load(f)
with open('../data/test_graphs_all.pkl', 'rb') as f:
    test_graphs = pickle.load(f)
with open('../data/val_graphs_all.pkl', 'rb') as f:
    val_graphs = pickle.load(f)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

model = HeteroGraph(2,
                    5,
                    10,
                    hidden_dims=20,
                    out_dims=3,
                    num_layers=2,
                    sage_aggr='sum',
                    batch_norm=True).to(device)

model.load_state_dict(torch.load('../models/model_l21e3_bn.pt', map_location="cpu", weights_only=False)) 

def test(model, test_graphs, criterion, device):
    model.eval()
    tmp_loss = 0
    tmp_acc = 0
    tmp_f1 = 0
    with torch.no_grad():
        for graph, _, _, _ in test_graphs:
            graph = graph.to(device)
            graph['sub'].y = graph['sub'].y.to(device)
            out = model(graph.x_dict, graph.edge_index_dict)
            loss = criterion(out, graph['sub'].y)
            tmp_loss += loss.item()

            pred = out.argmax(dim=1)
        
            correct = pred.eq(graph['sub'].y).sum().item()
            tmp_acc += correct / graph['sub'].y.size(0)

            tmp_f1 += f1_score(graph['sub'].y.cpu(), pred.cpu(), average='macro')

    total = len(test_graphs)
    return tmp_loss/total, tmp_acc/total, tmp_f1/total

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

print(test(model, test_graphs, criterion, device))