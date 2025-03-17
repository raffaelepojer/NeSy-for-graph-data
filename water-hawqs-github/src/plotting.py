import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.cm as cm

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

def plot_all_graphs(scenarios, data, num_bins, all_years):
    if all_years:
        fig, ax = plt.subplots(len(scenarios), len(list(range(2010, 2021))), figsize=(27, 16))
        years = list(range(2010, 2021))
        counter = 0
        for s, scenario in enumerate(scenarios):
            for yc, year in enumerate(years):
                for idx, (graph, G, y, m) in enumerate(data):
                    if graph.scenario == scenario and y == year:
                        main_nodes = list(range(1, graph['sub'].x.shape[0]+1))
                        sub_graph = G.subgraph(main_nodes)
                        ix = np.unravel_index(counter, ax.shape)
                        counter += 1
                        plot_ax(sub_graph, graph['sub'].y.cpu().detach().numpy(), f"{graph.scenario} {y}", num_bins, ax[ix], ix)

    else:
        fig, ax = plt.subplots(1, len(scenarios), figsize=(16, 4))
        counter = 0
        for s, scenario in enumerate(scenarios):
            for idx, (graph, G, y, m) in enumerate(data):
                if graph.scenario == scenario:
                    main_nodes = list(range(1, graph['sub'].x.shape[0]+1))
                    sub_graph = G.subgraph(main_nodes)
                    ix = np.unravel_index(counter, ax.shape)
                    counter += 1
                    plot_ax(sub_graph, graph['sub'].y.cpu().detach().numpy(), f"{graph.scenario} {y}", num_bins, ax[ix], ix)

    plt.show()    



def display_values_agr(data, agr_types):
    G = nx.DiGraph()

    for edge in data.edge_index_dict[('sub', 'to', 'sub')].t():
        G.add_node(edge[0].item(), node_type='sub')
        G.add_node(edge[1].item(), node_type='sub')
        G.add_edge(edge[0].item(), edge[1].item(), color='blue', weight=0.8, arrowsize=10)

    for edge in data.edge_index_dict[('hru_agr', 'to', 'sub')].t():
        G.add_node(edge[1].item()+23, node_type='hru_agr')
        G.add_edge(edge[0].item()+23, edge[1].item(), color='black', weight=0.1, arrowsize=3)


    main_nodes = range(data['sub'].x.shape[0])
    sub_graph = G.subgraph(main_nodes)
    pos = nx.spring_layout(sub_graph, seed=888, iterations=100)

    neighbors = {} 
    for node in main_nodes:
        neighbors[node] = []

    for node in G.nodes():
            if node not in main_nodes:
                for neighbor in G.neighbors(node):
                    neighbors[neighbor].append(node)
    radius = 0.08
    for node in main_nodes:
        num_neighbors = len(neighbors[node])
        for i, neighbor in enumerate(neighbors[node]):
            pos[neighbor] = [pos[node][0] + radius * np.cos(2 * np.pi * (i + 1) / num_neighbors), pos[node][1] + radius * np.sin(2 * np.pi * (i + 1) / num_neighbors)]

    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]
    arrow_sizes = [G[u][v]['arrowsize'] for u,v in edges]
    labels = nx.get_node_attributes(G, 'label')

    colors_ = cm.rainbow(np.linspace(0, 1, len(agr_types)))
    colors_map = {hru: colors_[i] for i, hru in enumerate(agr_types)}
    color_to_index = {c: i for i,c in enumerate(agr_types)}

    hru_colors = []
    for hru in data['hru_agr'].x:
        hru_colors.append(colors_map[agr_types[np.argmax(hru[:5]).item()]])

    for i in range(1, 24):
        hru_colors.insert(i-1, 'lightgrey')

    node_sizes = []
    for node in G.nodes:
        if int(node) in main_nodes:
            node_sizes.append(40)
        else:
            node_sizes.append(30*data['hru_agr'].x[int(node)-23][-1].item())
            

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=pos, with_labels=True, labels=labels, node_size=node_sizes, node_color=hru_colors, font_size=6, font_weight='light', edge_color=colors, width=weights, arrowsize=arrow_sizes)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in main_nodes}, font_size=6, font_weight='bold')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_[i], markersize=10, label=hru) for i, hru in enumerate(agr_types)], title='HRU', loc='upper right', fontsize='small')
    plt.axis('equal')
    plt.show()