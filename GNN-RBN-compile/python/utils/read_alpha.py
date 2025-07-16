from typing import List, Set, Tuple
import networkx as nx
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

def load_data(dataset: str,
              degree_as_node_label: bool = False,
              print_info: bool = True,
              graph_type: int = 2,
              undirected=True) -> Tuple[List[Data],
                                        Tuple[int, int, int]]:

    if graph_type == 1:
        raise NotImplementedError()

    print('Loading data...')
    graph_list: List[Data] = []

    unique_graph_label: Set[int] = set()
    unique_node_labels: Set[int] = set()
    unique_node_features: Set[int] = set()

    with open(dataset, 'r') as in_file:
        # first line correspond to the number of graphs
        n_graphs = int(in_file.readline().strip())
        for _ in range(n_graphs):
            n_nodes, graph_label = map(
                int, in_file.readline().strip().split(" "))

            # register graph label (not really important)
            unique_graph_label.add(graph_label)

            graph: nx.DiGraph = nx.DiGraph()
            node_labels: List[int] = []
            node_features: List[List[int]] = []

            # --- READING GRAPH ----
            # for each node in the graph
            for node_id in range(n_nodes):
                # add the index (starts at 0 to n_nodes-1)
                graph.add_node(node_id)

                # n_features, [features], node label, n_edges, [neighbors]
                # * only work for categorical node features
                node_row = list(
                    map(int, in_file.readline().strip().split(" ")))

                # ---- FEATURES ----
                # first comes the number of features
                n_features = node_row[0]
                # if n_features > max_features:
                #     max_features = n_features
                features = []
                if n_features != 0:
                    # get all features, column 1 to n_features-1
                    features = node_row[1:n_features + 1]
                    
                unique_node_features.update(features)

                node_features.append(features)
                # ---- /FEATURES ----

                # ---- LABELS ----
                # TODO: support multiple labels
                node_label = node_row[n_features + 1]
                unique_node_labels.add(node_label)

                node_labels.append(node_label)
                # ---- /LABELS ----

                # ---- EDGES ----
                # get the rest, the neighbours
                neighbors = node_row[n_features + 3:]
                # register connections
                for neighbor in neighbors:
                    graph.add_edge(node_id, neighbor)

                    if undirected:
                        graph.add_edge(neighbor, node_id)

                # ---- /EDGES ----
            # --- /READING GRAPH ----

            edges = torch.tensor(list(graph.edges), dtype=torch.long)
            node_labels = torch.tensor(node_labels)

            # placeholder
            features = torch.tensor(node_features)

            graph_list.append(
                Data(
                    x=features,
                    edge_index=edges.t().contiguous(),
                    node_labels=node_labels,
                    graph_label=torch.tensor([graph_label])
                ))

    num_features = len(unique_node_features)
    for graph in graph_list:
        # for multi-features
        if n_features > 1:
            graph_x = []
            # TODO better super dummy: remove the extra 0s in the fist one-hot label (alpha1)
            if n_features == 2:
                for (d1, d2) in graph.x:
                    x1 = d1.type(torch.FloatTensor)
                    x1 = torch.unsqueeze(x1, 0)
                    # x1 = torch.nn.functional.one_hot(d1.squeeze(), 2).type(torch.FloatTensor)
                    x2 = torch.nn.functional.one_hot(d2.squeeze(), num_features).type(torch.FloatTensor)
                    graph_x.append(torch.cat((x1, x2), 0))
            elif n_features == 3:
                for (d1, d2, d3) in graph.x:
                    x1 = d1.type(torch.FloatTensor)
                    x1 = torch.unsqueeze(x1, 0)
                    x2 = d2.type(torch.FloatTensor)
                    x2 = torch.unsqueeze(x2, 0)
                    x3 = torch.nn.functional.one_hot(d3.squeeze(), num_features).type(torch.FloatTensor)
                    graph_x.append(torch.cat((x1, x2, x3), 0))
            else:
                for i in range(graph.x.size()[0]):
                    tmp = []
                    for col in graph.x[i]:
                        x = torch.nn.functional.one_hot(col.squeeze(), num_features).type(torch.FloatTensor)
                        tmp.append(x)
                    tmp = torch.stack(tmp)
                    tmp = tmp.view(tmp.size()[0] * num_features)
                    graph_x.append(tmp)
                    
            graph.x = torch.stack(graph_x)
        else:
            x = torch.nn.functional.one_hot(
                    graph.x.squeeze(), num_features).type(torch.FloatTensor)
            graph.x = x

    if print_info:
        print(f"#Graphs: {len(graph_list)}")
        print(f"#Graphs Labels: {len(unique_graph_label)}")
        print(f"#Node Features: {len(unique_node_features)}")
        print(f"#Node Labels: {len(unique_node_labels)}")

    return graph_list, \
        (len(unique_graph_label),
         len(unique_node_features),
         len(unique_node_labels))


def separate_data(graph_list, seed: int,
                  test_size: float = 0.2):

    return train_test_split(
        graph_list,
        random_state=seed,
        test_size=test_size,
        shuffle=True)


if __name__ == "__main__":
    a, _ = load_data(dataset="../data/train-cycle-150-50-150.txt")
    print(a[0].x)
    print(a[0].edge_index)
    print(a[0].node_labels)
    print(a[0].graph_label)
