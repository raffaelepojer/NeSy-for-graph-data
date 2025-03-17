import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.cm as cm
import pickle
from model import *

def plot_ax(G, values, title, num_bins, ax, ix):
    max_value = int(max(values))
    min_value = int(min(values))
    set_values = list(range(min_value, max_value+1))
    color = cm.rainbow(np.linspace(0, 1, num_bins))
    node_colors = []
    for idx, value in enumerate(values):
        node_colors.append(color[int(value)])
    
    pos = nx.spring_layout(G, iterations=100, seed=5)
    nx.draw(G, pos=pos, with_labels=False, node_size=20, node_color=node_colors, edge_color='blue', width=0.8, arrowsize=10, ax=ax)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color[idx], label=value) for idx, value in enumerate(list(range(num_bins)))]
    if ix == (0, 0):
        ax.legend(handles=handles)
    # ax.axis('equal')
    ax.set_title(title)

base_path = "../scenarios/"
scenarios = ["default-2", "soy-3", "corn-3", "pasture-3", 'cosy-3', "pasture-corn", "soy-corn-2", "cosy-corn-2", "pasture-soy-2", "pasture-cosy-2", "cosy-soy-2",]
agr_types = ['CORN', 'COSY', 'PAST', 'SOYB']

with open('../data/train_graphs_new.pkl', 'rb') as f:
    train_graphs = pickle.load(f)
with open('../data/test_graphs_new.pkl', 'rb') as f:
    test_graphs = pickle.load(f)
with open('../data/val_graphs_new.pkl', 'rb') as f:
    val_graphs = pickle.load(f)

main_nodes = list(range(1, 24))
fig = plt.figure(); plt.clf()
fig, ax = plt.subplots(len(scenarios), 1, figsize=(4, 16))

model = HeteroGraph(2, 5, 10, 32, 3, 3)
model.eval()
model.load_state_dict(torch.load("../models/model.pt",  map_location="cpu", weights_only=False))

print(test_graphs+train_graphs+val_graphs)

years = list(range(2010, 2011))
counter = 0
for s, scenario in enumerate(scenarios):
    for yc, year in enumerate(years):
        for idx, (graph, G, y, m) in enumerate(test_graphs+train_graphs+val_graphs):
            if graph.scenario == scenario and y == year:
                sub_graph = G.subgraph(main_nodes)
                ix = np.unravel_index(counter, ax.shape)
                counter += 1
                out = model(graph.x_dict, graph.edge_index_dict)
                pred = out.argmax(dim=-1).cpu()
                plot_ax(sub_graph, pred.numpy(), f"{graph.scenario}", 3, ax[ix], ix)
    
plt.show()
