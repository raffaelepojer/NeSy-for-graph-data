from torch_geometric.datasets import WebKB, WikipediaNetwork, Planetoid
import networkx as nx
import torch
from torch_geometric.utils import to_networkx, from_networkx, subgraph
import numpy as np
import networkx as nx
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import torch_geometric.transforms as T
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from model import *
import pickle
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--scheduler', type=bool, default=False, help='Use scheduler')

parser.add_argument('--no_degree', action='store_false', default=True, help='do not use degree correction (degree correction only used with symmetric normalization)')
parser.add_argument('--no_sign', action='store_false', default=True, help='do not use signed weights')
parser.add_argument('--no_decay', action='store_false', default=True, help='do not use decaying in the residual connection')
parser.add_argument('--use_bn', action='store_true', default=False, help='use batch norm when not using decaying')
parser.add_argument('--use_ln', action='store_true', default=False, help='use layer norm when not using decaying')
parser.add_argument('--exponent', type=float, default=3.0, help='exponent in the decay function')

parser.add_argument('--row_normalized_adj', action='store_true', default=False, help='choose normalization')
parser.add_argument('--scale_init', type=float, default=0.5, help='initial values of scale (when decaying combination is not used)')
parser.add_argument('--deg_intercept_init', type=float, default=0.5, help='initial values of deg_intercept (when decaying combination is not used)')

parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--model', type=str, default="GCN", help='choose models: GCN, GGCN')   
parser.add_argument('--decay_rate', type=float, default=1.0, help='decay_rate in the decay function')    
parser.add_argument('--use_sparse', action='store_true', default=False, help='use sparse version of GGNN and GAT for large graphs')

args = parser.parse_args()
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")

def train_s(model, optimizer, data, scheduler=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device)).squeeze()
    preds = out.argmax(dim=1)
    acc = int((preds[data.train_mask] == data.y[data.train_mask].to(device)).sum()) / int(data.train_mask.sum())
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return acc, loss.item()

def validate_s(model, data):
    with torch.no_grad():
        model.eval()
        out = model(data.x.to(device), data.edge_index.to(device)).squeeze()
        pred = out.argmax(dim=1)
        acc = int((pred[data.val_mask] == data.y[data.val_mask].to(device)).sum()) / int(data.val_mask.sum())
        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask].to(device))
        return acc, loss.item()
    
def test_s(model, data, curr_mask):
    model.load_state_dict(torch.load(f'PATH/Heterophily_and_oversmoothing/pretrained/{args.model}_{args.data}_{curr_mask}.pt', weights_only=True))
    with torch.no_grad():
        model.eval()
        out = model(data.x.to(device), data.edge_index.to(device)).squeeze()
        pred = out.argmax(dim=1)
        acc = int((pred[data.test_mask] == data.y[data.test_mask].to(device)).sum()) / int(data.test_mask.sum())
        loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask].to(device))
        return acc, loss.item()

def train_test_single(model, epochs, graph_data, curr_mask, use_sparse=False, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scheduler = None

    best_val_loss = math.inf
    no_improvement_count = 0
    t = trange(epochs, desc="Stats: ", position=0)
    total_val_loss = []

    def sys_normalized_adjacency(adj):
        adj = sp.coo_matrix(adj)
        adj = adj + sp.eye(adj.shape[0])
        row_sum = np.array(adj.sum(1))
        row_sum=(row_sum==0)*1+row_sum
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def preprocess_features(features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        rowsum = (rowsum==0)*1+rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features
    
    def gen_graph_data(x, adj):
        data = Data(x=x, edge_index=adj)
        G = to_networkx(data)
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        adj = sys_normalized_adjacency(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        x = preprocess_features(x)
        x = torch.FloatTensor(x)
        if not use_sparse:
            adj = adj.to_dense()
        return Data(x=x, edge_index=adj, train_mask=graph_data.train_mask, val_mask=graph_data.val_mask, test_mask=graph_data.test_mask, y=graph_data.y)

    if model.__class__.__name__ == 'GGCN_raf':
        print('Using GGCN generated data')
        graph_data = gen_graph_data(graph_data.x, graph_data.edge_index)

    for epoch in t:
        train_acc, train_loss = train_s(model, optimizer, graph_data, scheduler)
        val_acc, val_loss = validate_s(model, graph_data)
        total_val_loss.append(val_loss)

        if verbose:
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train: {train_acc:.2f}, Val: {val_acc:.2f}, Val loss: {val_loss:.4f}")
        
        t.set_description('[Train_loss:{:.6f} Train_acc: {:.4f}, Val_acc: {:.4f}, Val_loss: {:.4f}]'.format(train_loss, train_acc, val_acc, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save(model.state_dict(), f'PATH/Heterophily_and_oversmoothing/pretrained/{args.model}_{args.data}_{curr_mask}.pt')
        else:
            no_improvement_count += 1

        if args.patience > 0 and no_improvement_count >= args.patience:
            print(f'Early stopping at epoch {epoch} due to no improvement in validation loss.')
            break
    
    test_acc, test_loss = test_s(model, graph_data, curr_mask)
    print(f'Best model test accuracy: {test_acc}')
    return test_acc

def compute_edge_homophily(pyg_graph, directed=True, mask=None):
    homophily = 0
    
    if mask is not None:
        nodes = torch.tensor(np.arange(0, pyg_graph.x.shape[0], 1)[mask])
        edge_index = subgraph(nodes, pyg_graph.edge_index)[0]
        graph_y = pyg_graph.y[nodes]
    else:
        graph_y = pyg_graph.y
        edge_index = pyg_graph.edge_index
        nodes = torch.tensor(np.arange(0, pyg_graph.x.shape[0], 1))

    
    for x, n in enumerate(nodes):
        edges = edge_index[:, edge_index[0] == n].t()
        for edge in edges:
            nbr = edge[1]                    
            if graph_y[(nodes==n).nonzero().item()] == graph_y[(nodes==nbr).nonzero().item()]:
                homophily += 1 if directed else 0.5
    if len(edge_index[0]) > 0:
        return homophily / len(edge_index[0])
    else:
        return 0


########################################################################################################################################################################
########################################################################################################################################################################

dataset_name = args.data
model_name = args.model

print(f'Device: {device}')

try:
    dataset_name = dataset_name.lower()
    dataset = torch.load(f'PATH/{dataset_name}/{dataset_name}.pt')
    pyg_data = dataset
except:
    print('Dataset not found')

G = to_networkx(pyg_data)
print(pyg_data)

num_classes = len(np.unique(pyg_data.y.numpy()))

train_graph = pyg_data.clone()
best_models = []
ts_accs = []
num_masks = 10
if len(pyg_data.train_mask.shape) > 1:
    num_masks = pyg_data.train_mask.shape[1]

with open(f'PATH/Heterophily_and_oversmoothing/pretrained/results_{model_name}_{dataset_name}.txt', 'w') as f:
    
    for i in range(num_masks):
        if dataset_name == 'cora':
            with np.load(f'PATH/Heterophily_and_oversmoothing/splits/{dataset_name}_split_0.6_0.2_{i}.npz') as splits_file:
                train_mask = splits_file['train_mask']
                train_mask = np.array(train_mask, dtype=bool)
                val_mask = splits_file['val_mask']
                val_mask = np.array(val_mask, dtype=bool)
                test_mask = splits_file['test_mask']
                test_mask = np.array(test_mask, dtype=bool)
                mask_tmp = train_mask + val_mask
                train_graph.train_mask = train_mask
                train_graph.test_mask = test_mask
                train_graph.val_mask = val_mask
        else:
            if len(pyg_data.train_mask.shape) == 1:            
                mask_tmp = pyg_data.train_mask + pyg_data.val_mask
            else:
                mask_tmp = pyg_data.train_mask.t()[i] + pyg_data.val_mask.t()[i]

            if len(pyg_data.train_mask.shape) > 1:
                train_graph.train_mask = pyg_data.train_mask.t()[i]
                train_graph.test_mask = pyg_data.test_mask.t()[i]
                train_graph.val_mask = pyg_data.val_mask.t()[i]
                print(f'Mask {i}')

        edge_homophily = compute_edge_homophily(pyg_data, G.is_directed(), mask_tmp)
        f.write(f'{i} - "Known" Homophily: {edge_homophily}\n')
        print(f'{i} - "Known" Homophily: {edge_homophily}')

        if model_name == 'GGCN':
            use_degree = (args.no_degree) & (not args.row_normalized_adj)
            use_sign = args.no_sign
            use_decay = args.no_decay
            use_bn = (args.use_bn) & (not use_decay)
            use_ln = (args.use_ln) & (not use_decay) & (not use_bn)
            model = GGCN_prim(nfeat=pyg_data.x.shape[1], nlayers=args.layer, nhidden=args.hidden, nclass=num_classes, dropout=args.dropout, decay_rate=args.decay_rate, exponent=args.exponent, use_degree=use_degree, use_sign=use_sign, use_decay=use_decay, use_sparse=args.use_sparse, scale_init=args.scale_init, deg_intercept_init=args.deg_intercept_init, use_bn=use_bn, use_ln=use_ln, generated=True, pre_features=True, primula=False).to(device)
        elif model_name == 'GCN':
            model = GraphNet(nfeat=pyg_data.x.shape[1], nlayers=args.layer, nhid=args.hidden, nclass=num_classes, dropout=args.dropout, primula=False).to(device)
            # model = GCN(nfeat=pyg_data.x.shape[1], nlayers=args.layer, nhid=args.hidden, nclass=num_classes, dropout=args.dropout).to(device)
        elif model_name == 'MLP':
            model = MLP(nfeat=pyg_data.x.shape[1], nlayers=args.layer, nhidden=args.hidden, nclass=num_classes, dropout=args.dropout, use_res=True).to(device)
        else:
            print('Model not found')
            exit(0)

        test_res = train_test_single(model=model, epochs=args.epochs, graph_data=train_graph, curr_mask=i, use_sparse=args.use_sparse, verbose=False)
        f.write(f'{i} - Test accuracy: {test_res}\n')
        print(model_name)
        ts_accs.append(test_res)

        f.write(f'----------------------------\n')
    
    print(f'Average test accuracy: {np.mean(ts_accs)}')
    print(f'Std test accuracy: {np.std(ts_accs)}')
    f.write(f'Average test accuracy: {np.mean(ts_accs)}\n')
    f.write(f'Std test accuracy: {np.std(ts_accs)}\n')
f.close()