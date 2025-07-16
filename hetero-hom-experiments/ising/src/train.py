import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
import math
from tqdm import trange
from torch_geometric.utils import to_networkx
from model import *


###### FUNCTION FOR GGCN ######
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

def gen_graph_data(graph_data, use_sparse=False):
    x = graph_data.x
    adj = graph_data.edge_index
    data = Data(x=x, edge_index=adj)
    G = to_networkx(data)
    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # x = preprocess_features(x)
    x = torch.FloatTensor(x)
    if not use_sparse:
        adj = adj.to_dense()
    return Data(x=x, edge_index=adj, train_mask=graph_data.train_mask, val_mask=graph_data.val_mask, test_mask=graph_data.test_mask, y=graph_data.y)
####################################

def train_s(model, optimizer, criterion, data, scheduler=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index).squeeze()
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item()

def test_s(model, criterion, data, generated=True):
    if model.__class__.__name__ == 'GGCN_raf' and not generated:
        print('Using GGCN generated data')
        data = gen_graph_data(data)

    with torch.no_grad():
        model.eval()
        out = model(data.x, data.edge_index).squeeze()
        pred = out.argmax(dim=1)
        losses = []
        accs = []
        
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            if int(mask.sum()) > 0:
                acc = int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())
                accs.append(acc)
                losses.append(criterion(out[mask], data.y[mask]))
            else:
                accs.append(0)
                losses.append(0)
        # return train_acc, val_acc, test_acc, val_loss, test_loss
        return accs, float(losses[1].detach().item()),  float(losses[2].detach().item())

def train_test_single(model, epochs, graph_data, model_path, patience=50, lr=0.01, weight_decay=0.01, use_sparse=False, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scheduler = None

    best_val_loss = math.inf
    best_test_acc = 0
    no_improvement_count = 0
    t = trange(epochs, desc="Stats: ", position=0)
    total_loss = []
    total_val_loss = []
    total_ts_acc = []

    if model.__class__.__name__ == 'GGCN_raf':
        print('Using GGCN generated data')
        graph_data = gen_graph_data(graph_data)

    for epoch in t:
        loss = train_s(model, optimizer, criterion, graph_data, scheduler)
        (train_acc, val_acc, ts_acc), val_loss, test_loss = test_s(model, criterion, graph_data)
        total_loss.append(test_loss)
        total_ts_acc.append(ts_acc)
        total_val_loss.append(val_loss)

        if verbose:
            print(f"Epoch: {epoch}, Train Loss: {loss:.4f}, Train: {train_acc:.2f}, Test: {ts_acc:.2f}, Val: {val_acc:.2f}, Val loss: {val_loss:.4f}")
        
        t.set_description('[Train_loss:{:.6f} Train_acc: {:.4f}, Test_acc: {:.4f}, Test_loss: {:.4f}]'.format(loss, train_acc, ts_acc, test_loss))

        if ts_acc > best_test_acc:
            best_test_acc = ts_acc

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save(model.state_dict(), model_path) 
        else:
            no_improvement_count += 1

        if patience > 0 and no_improvement_count >= patience:
            print(f'Early stopping at epoch {epoch} due to no improvement in validation loss.')
            epochs = epoch
            break
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # print(f'Best test acc found: {best_test_acc:.4f}')
    (train_acc, val_acc, ts_acc), val_loss, test_loss = test_s(model, criterion, graph_data)
    print(f'Last test accuracy: {ts_acc}')
    return ts_acc
    # plt.plot(total_loss, label='train loss')
    # plt.plot(total_ts_acc, label='test accuracy')
    # plt.plot(total_val_loss, label='validation loss', linestyle = 'dashed')
    # if model.__class__.__name__ == 'GGCN_raf':
    #     plt.title('GGCN')
    # elif model.__class__.__name__ == 'GraphNet':
    #     plt.title('GCN')
    # elif model.__class__.__name__ == 'MLP':
    #     plt.title('MLP')
    # plt.legend()
    # plt.show()