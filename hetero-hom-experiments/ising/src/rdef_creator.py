import numpy as np
from torch_geometric.utils import to_networkx, is_undirected, subgraph
import xml.etree.ElementTree as ET
from homph import *


def create_rdef_file(gdata, values, file_name, train_nodes, test_nodes, val_nodes, use_LP=False, use_HP=False):
    # known nodes are the nodes for which we want to have as known, all the others will be used to query
    # the train, test, and validation nodes are used for computing the accuracy in primula
    G = to_networkx(gdata, to_undirected=True)
    num_features_node = gdata.x.shape[1]
    
    values_string = ""
    for i, s in enumerate(values):
        if i < len(values)-1:
            values_string += s + ","
        else:
            values_string += s

    root = ET.Element("root")
    relations = ET.SubElement(root, "Relations")
    ET.SubElement(relations, "Rel", name="node", arity="1", argtypes="Domain", valtype="boolean", default="false", type="predefined", color="(240,240,240)")
    ET.SubElement(relations, "Rel", name="edge", arity="2", argtypes="node,node", valtype="boolean", default="false", type="predefined", color="(0,70,0)")
    
    ET.SubElement(relations, "Rel", name="CAT", arity="1", argtypes="node", valtype="categorical", values=values_string, default="?", type="probabilistic", color="(240,240,240)")

    for i in range(gdata.x.size()[1]):
        ET.SubElement(relations, "Rel", name=f"attr{i}", arity="1", argtypes="node", valtype="numeric", default="0.0", type="predefined", color="(240,240,240)")
    
    if use_LP or use_HP:
        ET.SubElement(relations, "Rel", name=f"hom", arity="1", argtypes="node", valtype="numeric", default="0.5", type="predefined", color="(40,40,40)")

    ET.SubElement(relations, "Rel", name="ground_POS", arity="1", argtypes="node", valtype="boolean", default="false", type="predefined", color="(240,0,0)")
    ET.SubElement(relations, "Rel", name="ground_NEG", arity="1", argtypes="node", valtype="boolean", default="false", type="predefined", color="(0,240,0)")
    # for el in values:
    #     ET.SubElement(relations, "Rel", name="ground_"+el, arity="1", argtypes="node", valtype="boolean", default="false", type="predefined", color="(240,240,240)")    
    
    ET.SubElement(relations, "Rel", name="train_nodes", arity="1", argtypes="node", valtype="boolean", default="false", type="predefined", color="(240,240,240)")
    ET.SubElement(relations, "Rel", name="test_nodes", arity="1", argtypes="node", valtype="boolean", default="false", type="predefined", color="(240,240,240)")
    ET.SubElement(relations, "Rel", name="val_nodes", arity="1", argtypes="node", valtype="boolean", default="false", type="predefined", color="(240,240,240)")
    ET.SubElement(relations, "Rel", name="query_nodes", arity="1", argtypes="node", valtype="boolean", default="false", type="predefined", color="(240,240,240)")

    # ET.SubElement(relations, "Rel", name="const", arity="2", argtypes="node,node", valtype="boolean", default="?", type="probabilistic", color="(0,50,0)")
    ET.SubElement(relations, "Rel", name="const", arity="1", argtypes="node", valtype="boolean", default="?", type="probabilistic", color="(0,50,0)")
    
    data_xml = ET.SubElement(root, "Data")
    data_for_input_domain = ET.SubElement(data_xml, "DataForInputDomain")
    domain = ET.SubElement(data_for_input_domain, "Domain")

    # query nodes = all - known
    known_nodes = np.concatenate((train_nodes, val_nodes))
    all_nodes = np.arange(0, gdata.x.size()[0], 1)
    query_nodes = np.setdiff1d(all_nodes, known_nodes)

    # pos = nx.spring_layout(G, k=0.4, seed=42, scale=800)
    cols = rows = 32
    scale = 50
    pos = {node: ((node // cols) * scale, (node % cols) * scale) for node in G.nodes}
    pred_nodes_str = ""
    query_nodes_s = ""
    train_nodes_s = ""; test_nodes_s = ""; val_nodes_s = ""
    for i, (node, (x, y)) in enumerate(pos.items()):
        min_x = min(coord[0] for coord in pos.values())
        min_y = min(coord[1] for coord in pos.values())

        ET.SubElement(domain, "obj", ind=str(i), name=str(i), coords="{x:.4f}".format(x=x)+","+"{y:.4f}".format(y=y))
        # ET.SubElement(domain, "obj", ind=str(i), name=str(i), coords="{x:.4f}".format(x=(x - min_x if min_x < 0 else x)+100)+","+"{y:.4f}".format(y=(y - min_y if min_y < 0 else y)+100))
        pred_nodes_str += f"({i})"
        if node in query_nodes:
            query_nodes_s += f"({i})" 
        if node in train_nodes:
            train_nodes_s += f"({i})" 
        elif node in test_nodes:
            test_nodes_s += f"({i})"
        elif node in val_nodes:
            val_nodes_s += f"({i})"
    
    predefined_rels = ET.SubElement(data_for_input_domain, "PredefinedRels")
    probabilistic_rels_case = ET.SubElement(data_for_input_domain, "ProbabilisticRelsCase")  

    # write the attributes for each classes
    for i in range(num_features_node):
        string_attr = ""
        for j, node in enumerate(G.nodes()):
            ET.SubElement(predefined_rels, "d", rel=f"attr{i}", args=f"({node})", val=f"{gdata.x[node][i].item()}")       

    if use_LP:
        # we use the ground truth for all the nodes (check)
        label_data = label_propagation(gdata.clone(), 2)
        for i in range(gdata.x.size()[0]):
            hom_k_g = 0.0
            count_k_g = 0
            for edges in gdata.edge_index[:, (gdata.edge_index[0] == i) | (gdata.edge_index[1] == i)].t():
                neighbor = edges[1] if edges[0] == i else edges[0]
                if label_data.y[neighbor].item() == label_data.y[i].item():
                # if gdata.y[neighbor].item() == gdata.y[i].item():
                    hom_k_g += 1
                count_k_g += 1
            if count_k_g > 0:
                hom_k_g = hom_k_g / count_k_g
            else:
                hom_k_g = 0

            ET.SubElement(predefined_rels, "d", rel=f"hom", args=f"({i})", val=f"{hom_k_g}")
    elif use_HP:
        local_hom = propagate_homphily3(gdata.clone(), iterations=1000, tolerance=0.0001)
        for i, hom in enumerate(local_hom):
            ET.SubElement(predefined_rels, "d", rel=f"hom", args=f"({i})", val=f"{local_hom[i]}")
                        

    ET.SubElement(predefined_rels, "d", rel="node", args=pred_nodes_str, val="true")
    ET.SubElement(predefined_rels, "d", rel="query_nodes", args=query_nodes_s, val="true")
    ET.SubElement(predefined_rels, "d", rel="train_nodes", args=train_nodes_s, val="true")
    ET.SubElement(predefined_rels, "d", rel="test_nodes", args=test_nodes_s, val="true")
    ET.SubElement(predefined_rels, "d", rel="val_nodes", args=val_nodes_s, val="true")
    
    for i, value in enumerate(values):
        known_nodes_s = ""
        groud_pos = ""
        for j, node in enumerate(G.nodes()):
            if node in known_nodes and gdata.y[node].item() == i:
                known_nodes_s += f"({j})"
            
            if gdata.y[node].item() == i:
                groud_pos += f"({node})" 
        
        ET.SubElement(probabilistic_rels_case, "d", rel="CAT", args=known_nodes_s, val=values[i])
        ET.SubElement(predefined_rels, "d", rel=f"ground_{values[i]}", args=groud_pos, val="true") 

    string_edges = ""
    string_const = ""

    processed_edges = set()
    for _, edge in enumerate(gdata.edge_index.t()):
        edge_tuple = (edge[0].item(), edge[1].item())
        reverse_edge_tuple = (edge[1].item(), edge[0].item())

        edg = f"({edge_tuple[0]},{edge_tuple[1]})"
        string_edges += edg
        string_const += edg + f"({edge_tuple[1]},{edge_tuple[0]})"
        
        # Check if the edge or its reverse has already been processed
        if edge_tuple not in processed_edges and reverse_edge_tuple not in processed_edges:
            # edg = f"({edge_tuple[0]},{edge_tuple[1]})"
            # string_edges += edg
            # string_const += edg + f"({edge_tuple[1]},{edge_tuple[0]})"
            
            # Add both the edge and its reverse to the set
            processed_edges.add(edge_tuple)
            processed_edges.add(reverse_edge_tuple)

    ET.SubElement(predefined_rels, "d", rel="edge", args=string_edges, val="true")
    # ET.SubElement(probabilistic_rels_case, "d", rel="const", args=string_edges, val="true")

    # here the constraint need to be applied to all the nodes! Not only on the "known nodes"
    if use_LP or use_HP:
        all_nodes_s = ""
        for node in all_nodes:
            all_nodes_s += f"({node})"
        ET.SubElement(probabilistic_rels_case, "d", rel="const", args=all_nodes_s, val="true")

    # write the file 
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(file_name, encoding='utf-8')