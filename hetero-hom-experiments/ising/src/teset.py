import torch
import numpy as np
import matplotlib.pyplot as plt
from model import *

settings = [(32, 0.5, 0.0, 0.4), (32, -0.5, 0.0, 0.4), (32, -0.4, 0.1, 0.4), (32, -0.7, 0.3, 0.4), (32, 0.9, 0.05, 0.4)]

for with_noise in [True, False]:
    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    for mm, m in enumerate(['GGCN', 'GCN', 'MLP']):
        counter = 0
        for ss, set in enumerate(settings):
            N = set[0]
            J = set[1]
            Jb = set[2]
            temp = set[3]

            data = None
            if with_noise:
                data = torch.load(f'PATH/ising/data/ising_data_{N}_{J}_{Jb}_{temp}_noisy.pt', weights_only=False)
            else:
                data = torch.load(f'PATH/ising/data/ising_data_{N}_{J}_{Jb}_{temp}.pt', weights_only=False)
            criterion = torch.nn.NLLLoss()
            accs = []
            model = None
            if m == 'GGCN':
                model = GGCN_raf(nfeat=data[4].x.shape[1], nlayers=2, nhidden=16, nclass=2, dropout=0.5, decay_rate=0.9, exponent=3.0, use_degree=True, use_sign=True, use_decay=True, use_sparse=False, scale_init=0.5, deg_intercept_init=0.5, use_bn=False, use_ln=False, generated=False, primula=True).to("cpu")
            elif m == 'GCN':
                model = GraphNet(nfeat=1, nlayers=2, nhid=16, nclass=2, dropout=0.3, primula=True).to("cpu")
            elif m == 'MLP':
                model = MLP(nfeat=1, nlayers=2, nhidden=32, nclass=2, dropout=0.5, use_res=True, primula=True).to("cpu")
            else:
                print('model not found')
                break
                
            model_path = ''
            if with_noise:
                model_path = f'PATH/ising/trained/{model.__class__.__name__}_{N}_{J}_{Jb}_{temp}_noisy_{4}_{0}.pt'
            else:
                model_path = f'PATH/ising/trained/{model.__class__.__name__}_{N}_{J}_{Jb}_{temp}_{4}_{0}.pt'

            model.load_state_dict(torch.load(model_path, weights_only=False))
            model.eval()
            out = model(data[4].x, data[4].edge_index).squeeze()
            predicted_class_gnn1 = out.argmax(dim=1)

            gnn_values = np.zeros(out.shape[0])
            for i in range(out.shape[0]):
                gnn_values[i] = out[i][predicted_class_gnn1[i]]
            gnn_values_2D = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    gnn_values_2D[i, j] = gnn_values[i*N+j]
            
            # ix = np.unravel_index(counter, axs.shape)
            axs[mm, ss].set_title(f'{m} H:{J} F:{Jb}')
            axs[mm, ss].imshow(gnn_values_2D, cmap='hot')
            plt.colorbar(axs[mm, ss].imshow(gnn_values_2D, cmap='hot'))
    plt.tight_layout()
    plt.show()