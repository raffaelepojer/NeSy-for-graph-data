from dataset_creator import from_swat_to_torch
from plotting import plot_all_graphs, display_values_agr
from model import HeteroGraph, EarlyStopper
import matplotlib.pyplot as plt
import torch
from torch import optim
import math
import random
import numpy as np
import pickle
from sklearn.metrics import f1_score

###########################################
## Train the GNN on the generated graphs ##
###########################################

if __name__ == '__main__':
    base_path = "../scenarios/"
    scenarios = ["default-2", "soy-3", "corn-3", "pasture-3", 'cosy-3', "cosy-corn-2", "soy-corn-2", "pasture-soy-2", "pasture-corn", "default-mix", "default-mix-2", "default-mix-3"]

    agr_types = ['CORN', 'COSY', 'PAST', 'SOYB']

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

    def train(model, train_graphs, optimizer, criterion, device):
        model.train()
        tmp_loss = 0
        tmp_acc = 0
        for idx, (graph, _, _, _) in enumerate(train_graphs):
            graph = graph.to(device)
            graph['sub'].y = graph['sub'].y.to(device)
            optimizer.zero_grad()
            out = model(graph.x_dict, graph.edge_index_dict)
            loss = criterion(out, graph['sub'].y)
            loss.backward()
            optimizer.step()
            tmp_loss += loss.item()
            pred = out.argmax(dim=1)
            correct = pred.eq(graph['sub'].y).sum().item()
            tmp_acc += correct / graph['sub'].y.size(0)
        
        return tmp_loss / len(train_graphs), tmp_acc / len(train_graphs)

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

    model = HeteroGraph(train_graphs[0][0]['sub'].x.shape[1],
                        train_graphs[0][0]['hru_agr'].x.shape[1],
                        train_graphs[0][0]['hru_urb'].x.shape[1],
                        hidden_dims=20,
                        out_dims=3,
                        num_layers=2,
                        sage_aggr='sum',
                        batch_norm=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    early_stopper = EarlyStopper(patience=40, min_delta=0.0)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs = []

    best_model = None
    best_loss = math.inf
    for epoch in range(450):
        train_loss, train_acc = train(model, train_graphs, optimizer, criterion, device)
        val_loss, val_acc, _ = test(model, val_graphs, criterion, device)
        test_loss, test_acc, test_f1 = test(model, test_graphs, criterion, device)
        scheduler.step()
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
        
        print(f"epoch: {epoch} - test loss/acc: {test_loss}/{test_acc} - test f1_macro: {test_f1}")

        if early_stopper.early_stop(val_loss):
            print("Exit from training before for early stopping at epoch: ", epoch)
            break

    print(f"Best loss: {best_loss}")
    
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.show()

    plt.plot(train_accs, label='train acc')
    plt.plot(test_accs, label='test acc')
    plt.legend()
    plt.show()
    print(f"mean test acc: {np.mean(test_accs)}")
    torch.save(best_model.state_dict(), '../models/model_l21e5_bn.pt')