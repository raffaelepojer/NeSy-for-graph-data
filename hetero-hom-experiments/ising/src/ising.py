import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
import pickle
import os

# https://rajeshrinet.github.io/blog/2014/ising-model/
class Ising():
    def __init__(self):
        self.graphs = []
        self.configurations = []
        self.energies = []
        self.N = None     
        self.J = None         
        self.Jb = None        
        self.temp = None  

    ''' Simulating the Ising model '''
    ## monte carlo moves
    def mcmove(self, config, N, J, Jb, mapping, beta):
        ''' This is to execute the monte carlo moves using 
        Metropolis algorithm such that detailed
        balance condition is satisified'''
        energy = 0     

        for _ in range(N**2):            
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s =  config[a, b]

            # nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
            nb = 0
            if a>0:
                nb += config[a-1,b]
            if a<N-1:
                nb += config[a+1,b]
            if b>0:
                nb += config[a,b-1]
            if b<N-1:
                nb += config[a,b+1]

            cost = J*s*nb + Jb*mapping(N, a, b)*s

            if cost < 0:	
                s *= -1
            elif rand() < np.exp(-cost*beta):
                s *= -1
            config[a, b] = s

            # calculate energy
            energy -= J*s*nb + Jb*s*mapping(N, a, b)

        return config, energy
    
    def simulate(self, N, J, Jb, temp=0.4, iterations_to_save=[1, 4, 32, 64, 512]):
        ''' This module simulates the Ising model'''
        self.N = N
        self.J = J
        self.Jb = Jb
        self.temp = temp
        config = 2*np.random.randint(2, size=(N,N))-1
        self.configurations.append((0, config.copy()))
        self.energies.append((0, 0))
        self.graphs.append((config.copy(), np.zeros((N,N))))

        def external_field(N, a, b):
            # add some random perturbation to the external field
            return -N + a + b # + np.random.normal(0, 40)

        b_val = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                b_val[i, j] = external_field(N, i, j)*Jb

        msrmnt = 512
        for i in range(msrmnt+1):
            _, energy = self.mcmove(config, N, J, Jb, external_field, 1.0/temp)
            if i in iterations_to_save:
                self.configurations.append((i, config.copy()))
                self.energies.append((i, energy))
                self.graphs.append((config.copy(), b_val.copy()))

    def grouped_plot(self, save=False):
        ''' Save a grouped plot of all saved configurations '''
        grouped_plot = plt.figure(figsize=(15, 15), dpi=80)
        for idx, (iteration, config) in enumerate(self.configurations):
            sp = grouped_plot.add_subplot(3, 3, idx + 1)
            sp.imshow(config)
            # sp.set_title(f'Iter: {iteration * self.N * self.N}, Energy: {self.energies[idx][1]:.2f}')
            sp.axis('off')
        if save:
            grouped_plot.savefig(f'ising/plots/grouped_{self.N}_{self.J}_{self.Jb}_{self.temp}.png', dpi=300)
        # plt.show()

    def single_plot(self, iteration_index, save_path=False):
        ''' Show a single plot for a specific configuration '''
        iteration, config = self.configurations[iteration_index]
        energy = self.energies[iteration_index][1]

        single_plot = plt.figure(figsize=(6, 6), dpi=100)
        plt.imshow(config)
        # plt.title(f'Iter: {iteration * self.N * self.N}, Energy: {energy:.2f}')
        plt.axis('off')
        # plt.axis('tight')
        if save_path is not None:
            single_plot.savefig(save_path, dpi=180, bbox_inches='tight', pad_inches = 0)
        # plt.show()
        
    def generateGraphs(self):
        nx_graphs = []
        for graph, b_val in self.graphs:
            G = nx.DiGraph()
            for i in range(graph.shape[0]):
                for j in range(graph.shape[1]):
                    node_id = i * graph.shape[1] + j
                    G.add_node(node_id, x=b_val[i, j], y=graph[i, j])
            
            for i in range(graph.shape[0]):
                for j in range(graph.shape[1]):
                    node_id = i * graph.shape[1] + j
                    if i > 0:
                        G.add_edge(node_id, node_id - graph.shape[1], x=b_val[i-1, j], y=graph[i-1, j])
                    if j > 0:
                        G.add_edge(node_id, node_id - 1, x=b_val[i, j-1], y=graph[i, j-1])
                    if i < graph.shape[0] - 1:
                        G.add_edge(node_id, node_id + graph.shape[1], x=b_val[i+1, j], y=graph[i+1, j])
                    if j < graph.shape[1] - 1:
                        G.add_edge(node_id, node_id + 1, x=b_val[i, j+1], y=graph[i, j+1])
            
            nx_graphs.append(G)
        return nx_graphs


def edge_homophily(pyg_data):
    edge_index = pyg_data.edge_index
    y = pyg_data.y
    num_same = 0
    num_edges = edge_index.size(1)
    for i in range(num_edges):
        src_label = y[edge_index[0, i]].item()
        trg_label = y[edge_index[1, i]].item()
        if src_label == trg_label:
            num_same += 1
    return num_same / num_edges

def count_classes(G, split=None):
    class_counts = {}
    for node in G.nodes():
        if split is None:
            if G.nodes[node]['y'] not in class_counts:
                class_counts[G.nodes[node]['y']] = 0
            class_counts[G.nodes[node]['y']] += 1
        else:
            if node in split:
                if G.nodes[node]['y'] not in class_counts:
                    class_counts[G.nodes[node]['y']] = 0
                class_counts[G.nodes[node]['y']] += 1
    return class_counts


def create_splits(num_nodes, N, train_ratio=0.48, val_ratio=0.32):
    indices = np.random.permutation(num_nodes - 2) + 1  # exclude the first and last nodes
    
    train_size = int(train_ratio * (num_nodes - 2))
    val_size = int(val_ratio * (num_nodes - 2))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    masks = {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }

    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(base_path, '..')
    mask_path = os.path.join(base_path, 'masks', 'mask.pkl')
    plots_path = os.path.join(base_path, 'plots')
    with open(mask_path, 'wb') as f:
        pickle.dump(masks, f)
    
    train_nodes = np.arange(0, num_nodes, 1)[train_mask]
    val_nodes = np.arange(0, num_nodes, 1)[val_mask]
    ising = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ising[i, j] = 1.0 if (i*N+j in train_nodes) or (i*N+j in val_nodes) else 0.0
    
    print(ising)
    plt.figure(figsize=(4,4), dpi=80)
    plt.imshow(ising, cmap='binary', interpolation='none')
    plt.axis('tight')
    plt.axis('off')
    plt.savefig(f'{plots_path}/train_val_mask.png', bbox_inches='tight', pad_inches = 0)

    return train_mask, val_mask, test_mask
    
def graph_to_data(G):
    data = from_networkx(G)
    data.x = torch.tensor([[node[1]['x']] for node in G.nodes(data=True)], dtype=torch.float)
    data.y = torch.tensor([node[1]['y'] if node[1]['y']==1 else 0 for node in G.nodes(data=True)], dtype=torch.long)

    # PATH - FILE mask.pkl !
    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(base_path, '..')
    mask_path = os.path.join(base_path, 'masks', 'mask.pkl')
    if os.path.exists(mask_path):
        with open(mask_path, 'rb') as f:
            masks = pickle.load(f)
        data.train_mask = masks['train_mask']
        data.val_mask = masks['val_mask']
        data.test_mask = masks['test_mask']
    else:
        data.train_mask, data.val_mask, data.test_mask = create_splits(data.num_nodes, 32)
    
    return data