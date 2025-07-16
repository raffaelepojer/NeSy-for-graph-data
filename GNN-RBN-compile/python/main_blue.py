import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import utils.read_alpha
import math
from gnn import *
from datetime import datetime
from utils.early_stopper import EarlyStopper
from rbn.gnn_to_rbn import *
from utils.utils import *
from argparse import ArgumentParser 
from torch.utils.tensorboard import SummaryWriter
import pickle
import time

def train(model, data, criterion, optimizer, device="cpu"):
    model.train()
    t = 0  # count of true predictions
    f = 0  # count of false predictions
    epoch_loss = 0
    for batch in data:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        target = batch.node_labels
        
        out = torch.squeeze(out)
        l = criterion(out, target.float())
        
        pred = (out > 0.5).long()

        tt = len(np.where(target.cpu() == pred.cpu())[0])
        t += tt
        f += len(target)-tt

        epoch_loss += l.detach().item()
        l.backward()
        optimizer.step()
    accuracy = t/(t+f)
    avg_loss = epoch_loss/len(data)
    return avg_loss, epoch_loss, accuracy

def test(model, data, criterion, device="cpu"):
    model.eval()
    t = 0
    f = 0
    epoch_loss = 0
    for batch in data:
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
        target = batch.node_labels
        
        out = torch.squeeze(out)
        l = criterion(out, target.float())

        epoch_loss += l.detach().item()
        
        pred = (out > 0.5).long()

        tt = len(torch.where(target == pred)[0])
        t += tt
        f += len(target)-tt

    accuracy = t/(t+f)
    avg_loss = epoch_loss / len(data)
    return avg_loss, accuracy

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(args):
    # seed_everything(args.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # ACR-GNN definition
    hidden_dim = args.hidden_dim
    gnn_layers = len(hidden_dim)
    # feature_dim = 7
    epochs = args.epochs
    final_readout = args.final_readout
    lr = args.lr
    min_acc = 0.0

    BATCH_SIZE = args.batch_size

    ds_name = args.ds_name
    save_logs = args.save_logs
    save_rbn = args.save_rbn
    alpha = 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # data_dir = os.path.join(script_dir, '..', 'datasets', 'alpha', f"p{alpha}_a_small")
    data_dir = os.path.join(script_dir, '..', 'datasets', 'alpha', f"p{alpha}")


    # original dataset
    data_stem_train = "train-random-erdos-5000-40-50.txt" 
    data_stem_test_1 = "test-random-erdos-500-40-50.txt"
    data_stem_test_2 = "test-random-erdos-500-51-60.txt"

    # data_stem_train = "train-barabasi-m2_2-5000-5-8.txt" 
    # data_stem_test_1 = "test-barabasi-m2_2-500-5-8.txt"
    # data_stem_test_2 = "test-barabasi-m2_2-500-10-15.txt"
    
    
    data_train, _ = utils.read_alpha.load_data(
                        dataset=os.path.join(data_dir, data_stem_train),
                        degree_as_node_label=False)
    data_test_1, _ = utils.read_alpha.load_data(
                        dataset=os.path.join(data_dir, data_stem_test_1),
                        degree_as_node_label=False)
    data_test_2, _ = utils.read_alpha.load_data(
                        dataset=os.path.join(data_dir, data_stem_test_2),
                        degree_as_node_label=False)

    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader_1 = DataLoader(data_test_1, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_2 = DataLoader(data_test_2, batch_size=BATCH_SIZE, shuffle=False)

    # get the feature dimension
    feature_dim = next(iter(train_loader)).x.shape[1]
    print("feature dim: {}".format(feature_dim))
    print("Number of batches test 1: {}".format(len(test_loader_1)))

    gnn_layers_string = print_list_with_underscores(hidden_dim)
    experiment_path = os.path.join(script_dir, '..', 'models', f"{ds_name}_{gnn_layers_string}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    file_stem ="RBN_acr_graph_" + ds_name
    tensorboard_path = os.path.join(experiment_path, "tensorboard")

    if save_rbn:
        create_folder_if_not_exists(experiment_path)
    if save_logs:
        create_folder_if_not_exists(tensorboard_path)
        writer = SummaryWriter(log_dir=tensorboard_path)

    print("testing acr-gnn with {} layers".format(gnn_layers))
    history = {'train_loss': [], 'test_loss': [],
               'train_acc': [], 'test_acc': []}

    best_loss = math.inf
    model = MYACRGnnNode(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=gnn_layers,            
        final_read=final_readout,
        num_classes=1
    )
    model.to(device)    
    early_stopper = EarlyStopper(patience=25, min_delta=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss = nn.BCEWithLogitsLoss(reduction='mean')
    loss = nn.BCELoss(reduction='sum')

    epoch_losses = []
    total_time = 0
    num_restarts = 5
    for r in range(num_restarts):
        model.init_model()
        model.train()
        
        epoch_losses.append([])
        for epoch in range(epochs):
            
            print("lr:", optimizer.param_groups[0]['lr'])
            
            start_time = time.time()
            train_loss, epoch_loss, train_acc = train(model, train_loader, loss, optimizer, device)
            end_time = time.time()
            total_time += (end_time - start_time)

            train_loss, epoch_loss, train_acc = train(model, test_loader_1, loss, optimizer, device)

            test_loss, test_acc = test(model, test_loader_2, loss, device)

            print("epoch loss:", epoch_loss)
            epoch_losses[r].append(epoch_loss)

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

            if early_stopper.early_stop(test_loss):
                print("Exit from training before for early stopping")
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_epoch = epoch
                    best_params = model.state_dict()
                break

            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch
                best_params = model.state_dict()

            if save_logs:
                writer.add_scalars('runs_{}'.format(final_readout), {
                    'Loss/train': train_loss, 'Loss/test': test_loss}, epoch+1)
                writer.add_scalars('runs_{}'.format(final_readout), {
                    'Acc/train': train_acc, 'Acc/test': test_acc}, epoch+1)

            print("Epoch: {}".format(epoch+1))
            print("\tTrain : \tTrain loss: {:.4f}\tTrain accuracy: {:.2f}".format(
                train_loss, train_acc))
            print("\tTest: \tTest loss:  {:.4f}\tTest accuracy:  {:.2f}".format(
                test_loss, test_acc))
            
            
    print("total time: ", total_time)
    avg_time = (total_time / (num_restarts*epochs))
    print(f"Average time per epoch: {avg_time} seconds.")

    # with open(os.path.join("/Users/lz50rg/Dev/NeSy-for-graph-data/GNN-RBN-compile/", "epoch_losses_big.pkl"), "wb") as f:
    #     pickle.dump(epoch_losses, f)

    for p in model.parameters():
        print(p)

    print("Best model found at epoch {} with loss {}".format(best_epoch+1, best_loss))

    print("*"*40)
    print("final train accuracy:\t{}".format(train_acc))
    model.load_state_dict(best_params)
    model.eval()
    val_loss, val_acc = test(model, test_loader_2, loss, device)
    print("final val accuracy:\t{}".format(val_acc))
    print("*"*40)
    
    avg_train_loss = np.mean(history['train_loss'])
    avg_test_loss = np.mean(history['test_loss'])
    avg_train_acc = np.mean(history['train_acc'])
    avg_test_acc = np.mean(history['test_acc'])

    print("Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Test Acc: {:.3f}".format(
        avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc))
    print("*"*40)

    # export model
    if test_acc > min_acc and save_rbn:
        feature_names = ["color"]
        val_feat = ["blue", "green", "red", "yellow", "purple"]
        feature_probs = [1]

        base_name = f"{file_stem}" + "_" + \
            f"{print_list_with_underscores(hidden_dim)}"
        rbn_name = os.path.join(experiment_path, base_name + ".rbn")
        gnn_weights = os.path.join(experiment_path, base_name + ".pt")
        write_rbn_ACR_node(rbn_name, model, ds_name, feature_names, feature_probs,
                            constraints=args.constraints, soft_prob=0.99, include_edge=0.0, categorical=val_feat, read_type=final_readout)

        rbn_name = os.path.join(experiment_path, base_name + "_param.rbn")
        write_rbn_ACR_node(rbn_name, model, ds_name, feature_names, feature_probs,
                            constraints=args.constraints, soft_prob=0.99, include_edge=0.0, categorical=val_feat,read_type=final_readout, write_param=True)
        torch.save(model.state_dict(), gnn_weights)
        print("Files written to: ", experiment_path)

    if save_logs:
        writer.flush()
        writer.close()

if __name__ == "__main__":
    parser = ArgumentParser(description="ACR-GNN for blue experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--hidden_dim", nargs="+", type=int, default=[5], help="Hidden layer dimensions")
    parser.add_argument("--final_readout", type=str, default="add", help="Final readout function")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--constraints", type=bool, default=False, help="Constraints for RBN")
    parser.add_argument("--ds_name", type=str, default="alpha1", help="Dataset name")
    parser.add_argument("--save_logs", type=bool, default=False, help="Save logs")
    parser.add_argument("--save_rbn", type=bool, default=True, help="Save RBN")
    args = parser.parse_args()
    main(args)