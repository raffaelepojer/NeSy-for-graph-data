
import torch
import numpy as np
from torch_geometric.utils import to_undirected

def label_propagation(data, num_class, iterations=100, tolerance=0.0001):
    # count train/val labels
    dict_label = dict.fromkeys(list(range(num_class)), 0)
    total_know = data.train_mask.sum() + data.val_mask.sum()
    for i, label in enumerate(data.y):
        if data.train_mask[i] or data.val_mask[i]:
            dict_label[label.item()] += 1

    propagated_node = np.zeros((data.y.shape[0], num_class))

    # init vector for test labels
    prop_score = np.zeros(num_class)
    for i in range(num_class):
        prop_score[i] = dict_label[i] / total_know

    # init label propagation
    for i, label in enumerate(data.y):
        if data.train_mask[i] or data.val_mask[i]:
            propagated_node[i] = torch.nn.functional.one_hot(label, num_classes=num_class)
        if data.test_mask[i]:
            propagated_node[i] = torch.tensor(prop_score)

    def get_neighbors(graph, node):
        # return graph.edge_index[1, graph.edge_index[0] == node]
        return torch.cat((graph.edge_index[1, graph.edge_index[0] == node], graph.edge_index[0, graph.edge_index[1] == node]))

    stabilized = np.zeros(data.y.shape[0], dtype=bool)

    for iter in range(iterations):
        for i, label in enumerate(data.y):
            if data.test_mask[i]:
                prev_vals = propagated_node[i].copy()
                neighs = get_neighbors(data, i)
                if len(neighs) == 0:
                    propagated_node[i] = torch.tensor(prop_score)
                else:
                    for j in range(num_class):
                        sum_n = 0
                        for neigh in neighs:
                            sum_n += propagated_node[neigh.item()][j]
                        propagated_node[i][j] = sum_n / len(neighs)

                change = np.linalg.norm(prev_vals - propagated_node[i])
                if change < tolerance:
                    stabilized[i] = True

        if np.all(stabilized[data.test_mask]):
            print(f"Converged at iteration {iter + 1}")
            break

    # assign each value with arg max the value
    new_y = np.zeros((len(data.y)))
    for i, vals in enumerate(propagated_node):
        max_val = np.max(vals)
        max_indices = np.where(vals == max_val)[0]

        if len(max_indices) == 1:  # Single maximum value
            new_y[i] = max_indices[0]
        else: 
            new_y[i] = np.random.choice(max_indices)
    data.y = torch.tensor(new_y, dtype=torch.long)   
    return data

def propagate_homphily3(data, iterations=100, tolerance=0.0001):        
    data.edge_index = to_undirected(data.edge_index)
    hom = np.zeros(data.y.shape[0])
    train_hom  = np.zeros(data.y.shape[0])

    all_nodes =  np.arange(0, data.x.size()[0])
    train_nodes = np.arange(0, data.x.size()[0])[(data.train_mask | data.val_mask)]
    train_mask = data.train_mask | data.val_mask
    test_nodes = np.arange(0, data.x.size()[0])[~(data.train_mask | data.val_mask)]

    # compute local homophily on the training nodes
    nodes_to_init = [] # save the nodes that need to be assigned with the init values
    for i in range(data.x.size()[0]):
        if train_mask[i]:
            num_neigh = 0; hom[i]=0
            for edge in data.edge_index[:, data.edge_index[0] == i].t():
                nbr = edge[1].item()
                if train_mask[nbr]:
                    if data.y[edge[1]].item() == data.y[i].item():
                        hom[i] += 1
                    num_neigh += 1
            if num_neigh > 0:
                hom[i] = hom[i] / num_neigh
                train_hom[i] = hom[i]
            else:
                nodes_to_init.append(i)
    
    init_value = train_hom.sum() / (len(train_nodes)-len(nodes_to_init))
    print("avg train-edge-hom: ", init_value)
    # fill the test nodes with the initial values (avg train hom)
    hom[test_nodes] = init_value
    hom[nodes_to_init] = init_value
    
    # propagate the homophily on the node which is not possible to compute directly the hom value
    converged = False
    for iter in range(iterations):
        converged = True
        for node in range(data.x.size()[0]):
            hom_prev = hom[node]
            neigh = [edge[1].item() for edge in data.edge_index[:, data.edge_index[0] == node].t()]
            train_neigh = np.intersect1d(neigh, train_nodes)
            test_neigh = np.intersect1d(neigh, test_nodes)

            # if node is test or a train with all test nodes as neighbor
            if node in test_nodes or (node in train_nodes and len(train_neigh)==0):
                hom[node] = 0; 
                for nbr in neigh:
                    hom[node] += hom[nbr]
                hom[node] = hom[node]/len(neigh) if len(neigh) > 0 else init_value
            
            # if node is a train node with at least one test node as neighbor
            elif node in train_nodes and len(test_neigh) > 0:
                avg_test_hom = 0
                for nbr in test_neigh:
                    avg_test_hom += hom[nbr]
                avg_test_hom /= len(test_neigh)
                hom[node] = (train_hom[node]*len(train_neigh) + avg_test_hom*len(test_neigh)) / (len(train_neigh)+len(test_neigh))
            if abs(hom_prev - hom[node]) > tolerance:
                converged = False
        
        assert hom[i]>=0 and hom[i]<=1, f"invalid hom: {hom[i]}!!"  
        if converged:
            print(f"2 - Converged at iteration {iter + 1}")
            break

    return hom
