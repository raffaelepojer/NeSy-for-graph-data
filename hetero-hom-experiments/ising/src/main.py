from ising import *
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import math
import random

from model import *
from train import *
from homph import *
from rdef_creator import *


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# settings = [(32, 0.5, 0.0, 0.4)]
# settings = [(32, -0.5, 0.0, 0.4)]
# settings = [(32, -0.4, 0.1, 0.4)]
# settings = [(32, -0.7, 0.3, 0.4)]
# settings = [(32, 0.9, 0.05, 0.4)]

settings = [(32, 0.5, 0.0, 0.4), (32, -0.5, 0.0, 0.4), (32, -0.4, 0.1, 0.4), (32, -0.7, 0.3, 0.4), (32, 0.9, 0.05, 0.4)]

for set in settings:

    # GENERATE GRAPH #########################################################################################
    N = set[0]
    J = set[1]
    Jb = set[2]
    temp = set[3]

    rm = Ising()
    rm.simulate(N, J, Jb, temp, iterations_to_save=[1, 4, 32, 64, 512])
    rm.single_plot(4, save_path=f'/Users/lz50rg/Documents/papers/logic-constraint-gnn-paper/ising/{N}_{J}_{Jb}_{temp}_{4}.png')

    nx_graphs = rm.generateGraphs()
    data = []
    for i, graph in enumerate(nx_graphs):
        pyd = graph_to_data(graph)
        print(f"edge-h [{i}]: {edge_homophily(pyd)}")
        data.append(pyd)

    torch.save(data, f'PATH/ising/data/ising_data_{N}_{J}_{Jb}_{temp}.pt')

    i = 4
    pyg_data = data[i]

    for i, graph in enumerate(nx_graphs):
        count = count_classes(graph, None)
        print(f'all [{i}] 1:{count[1]/(count[1]+count[-1])}\t0:{count[-1]/(count[1]+count[-1])}')

    t_mask = pyg_data.train_mask + pyg_data.val_mask
    train_nodes = np.arange(0, pyg_data.x.shape[0], 1)[t_mask]
    for i, graph in enumerate(nx_graphs):
        count = count_classes(graph, train_nodes)
        print(f'train [{i}] 1:{count[1]/(count[1]+count[-1])}\t0:{count[-1]/(count[1]+count[-1])}')

    test_nodes = np.arange(0, pyg_data.x.shape[0], 1)[pyg_data.test_mask]
    for i, graph in enumerate(nx_graphs):
        count = count_classes(graph, test_nodes)
        print(f'test [{i}] 1:{count[1]/(count[1]+count[-1])}\t0:{count[-1]/(count[1]+count[-1])}')

    # PLOT FEATURES #########################################################################################

    maxv=-math.inf; minv=math.inf
    features_2D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            features_2D[i, j] = data[4].x[i*N+j][0].item()
            maxv = max(data[4].x[i*N+j][0].item(), maxv)
            minv = min(data[4].x[i*N+j][0].item(), minv)

    print(maxv, minv)

    plt.figure(figsize=(4,4), dpi=80)
    plt.imshow(features_2D, cmap='coolwarm', vmin=-9.6, vmax=9.0)
    plt.axis('tight')
    plt.axis('off')
    plt.show()

    # ADD NOISE #########################################################################################

    features_2D_noisy = np.zeros((N, N))
    noisy_data = copy.deepcopy(data)
    for i in range(N):
        for j in range(N):
            val = np.random.normal(0, N*abs(Jb))
            features_2D_noisy[i, j] = data[4].x[i*N+j][0].item() + val
            for k in range(len(data)):
                noisy_data[k].x[i*N+j][0] = data[k].x[i*N+j][0].item() + val

    plt.figure(figsize=(4,4), dpi=80)
    plt.imshow(features_2D_noisy, cmap='coolwarm', vmin=-9.6, vmax=9.0)
    plt.axis('tight')
    plt.axis('off')
    torch.save(noisy_data, f'PATH/ising/data/ising_data_{N}_{J}_{Jb}_{temp}_noisy.pt')

    # COMPUTE HOMPHILY #########################################################################################

    one_data = data[4].clone()
    propagated_hom = propagate_homphily3(one_data.clone(), iterations=1000, tolerance=0.0001)

    hom_2D = np.zeros((N, N))
    for i in range(one_data.x.size()[0]):
        hom_2D[i // N, i % N] = propagated_hom[i]

    plt.figure(figsize=(4,4), dpi=80)
    plt.imshow(hom_2D, cmap='coolwarm')
    plt.axis('off')
    plt.axis('tight')
    plt.show()


    # SAVE RDEF #########################################################################################

    for s in [True, False]:
        if s:
            data = torch.load(f'PATH/ising/data/ising_data_{N}_{J}_{Jb}_{temp}_noisy.pt', weights_only=False)
        else:
            data = torch.load(f'PATH/ising/data/ising_data_{N}_{J}_{Jb}_{temp}.pt', weights_only=False)

        # write the RDEF file with homophily propagation
        train_nodes = np.arange(0, pyg_data.x.shape[0], 1)[pyg_data.train_mask]
        test_nodes = np.arange(0, pyg_data.x.shape[0], 1)[pyg_data.test_mask]
        val_nodes = np.arange(0, pyg_data.x.shape[0], 1)[pyg_data.val_mask]
        rdef_path = f"PATH/ising/rdef/ising_{N}_{J}_{Jb}_{temp}_{i}"
        if s:
            rdef_path += "_nodeconst_HP_noisy"
        else:   
            rdef_path += "_nodeconst_HP"
        
        rdef_path += ".rdef"
        create_rdef_file(pyg_data, ["POS", "NEG"], rdef_path, train_nodes, test_nodes, val_nodes, use_LP=False, use_HP=True)
        print(f"RDEF file created: {rdef_path}")

    # TRAIN GNNs #########################################################################################
    with open(f'PATH/ising/models_results.txt', 'a') as f:
        f.write(f'{set}\n')
        for data, with_noise in [(torch.load(f'PATH/ising/data/ising_data_{N}_{J}_{Jb}_{temp}_noisy.pt', weights_only=False), True), (torch.load(f'PATH/ising/data/ising_data_{N}_{J}_{Jb}_{temp}.pt', weights_only=False), False)]:
            print('***'*30)
            f.write(f'with noise: {with_noise}\n')
            for m in ['GGCN', 'GCN', 'MLP']:
                accs = []
                for restart in range(5):
                    model = None
                    if m == 'GGCN':
                        print('Using GGCN')
                        model = GGCN_raf(nfeat=data[4].x.shape[1], nlayers=2, nhidden=16, nclass=2, dropout=0.5, decay_rate=0.9, exponent=3.0, use_degree=True, use_sign=True, use_decay=True, use_sparse=False, scale_init=0.5, deg_intercept_init=0.5, use_bn=False, use_ln=False, generated=True).to("cpu")
                    elif m == 'GCN':
                        print('Using GraphGCN')
                        model = GraphNet(nfeat=1, nlayers=2, nhid=16, nclass=2, dropout=0.3, primula=False).to("cpu")
                    elif m == 'MLP':
                        print('Using MLP')
                        model = MLP(nfeat=1, nlayers=2, nhidden=32, nclass=2, dropout=0.5, use_res=True).to("cpu")
                    else:
                        print('model not found')
                        break
                        
                    model_path = ''
                    if with_noise:
                        print('WITH NOISE')
                        model_path = f'PATH/ising/trained/{model.__class__.__name__}_{N}_{J}_{Jb}_{temp}_noisy_{4}_{restart}.pt'
                    else:
                        print('WITHOUT NOISE')
                        model_path = f'PATH/ising/trained/{model.__class__.__name__}_{N}_{J}_{Jb}_{temp}_{4}_{restart}.pt'

                    test_acc = train_test_single(model, 400, data[4], model_path=model_path, patience=50, lr=0.01, weight_decay=0.01, verbose=False)
                    accs.append(test_acc)
                
                print(f'{m} mean:{np.mean(accs)} std:{ np.std(accs)}')
                f.write(f'{m} mean:{np.mean(accs)} std:{np.std(accs)}\n')
        
        print('***'*30)


        