import torch
import numpy as np
import networkx as nx
import xml.etree.ElementTree as ET
from torch_geometric.utils import to_networkx
from torch_geometric.data import HeteroData
import pickle
import torch.nn as nn
import os 

##################################################################
## Generate the .rdef file (data) for Primula from a graph data ##
##################################################################

def create_rdef_file(all_data, with_labels, file_name):
    data = all_data[0][0]
    G = all_data[0][1]
    val_luse = ['CORN', 'COSY', 'PAST', 'SOYB']
    val_urb = ['BERM', 'FESC', 'FRSD', 'FRST', 'RIWF', 'RIWN', 'UPWF', 'UPWN', 'WATR']
    
    # create the string for the values in rdef
    val_luse_string = ""
    val_urb_string = ""
    for i, s in enumerate(val_luse):
        if i < len(val_luse)-1:
            val_luse_string += s + ","
        else:
            val_luse_string += s

    for i, s in enumerate(val_urb):
        if i < len(val_urb)-1:
            val_urb_string += s + ","
        else:
            val_urb_string += s
    val_pollution = ['LOW', 'MED', 'HIG']

    root = ET.Element("root")
    relations = ET.SubElement(root, "Relations")
    ET.SubElement(relations, "Rel", name="sub", arity="1", argtypes="Domain", valtype="boolean", default="false", type="predefined", color="(100,100,100)")
    ET.SubElement(relations, "Rel", name="hru_agr", arity="1", argtypes="Domain", valtype="boolean", default="false", type="predefined", color="(100,100,100)")
    ET.SubElement(relations, "Rel", name="hru_urb", arity="1", argtypes="Domain", valtype="boolean", default="false", type="predefined", color="(100,100,100)")
    
    ET.SubElement(relations, "Rel", name="sub_to_sub", arity="2", argtypes="sub,sub", valtype="boolean", default="false", type="predefined", color="(0,70,0)")
    ET.SubElement(relations, "Rel", name="hru_agr_to_sub", arity="2", argtypes="hru_agr,sub", valtype="boolean", default="false", type="predefined", color="(0,70,0)")
    ET.SubElement(relations, "Rel", name="sub_to_hru_agr", arity="2", argtypes="sub,hru_agr", valtype="boolean", default="false", type="predefined", color="(0,70,0)")
    ET.SubElement(relations, "Rel", name="hru_urb_to_sub", arity="2", argtypes="hru_urb,sub", valtype="boolean", default="false", type="predefined", color="(0,70,0)")
    ET.SubElement(relations, "Rel", name="sub_to_hru_urb", arity="2", argtypes="sub,hru_urb", valtype="boolean", default="false", type="predefined", color="(0,70,0)")
    
    ET.SubElement(relations, "Rel", name="LandUse", arity="1", argtypes="hru_agr", valtype="categorical", values=val_luse_string, default="?", type="probabilistic", color="(180,20,20)")
    ET.SubElement(relations, "Rel", name="LandUseUrb", arity="1", argtypes="hru_urb", valtype="categorical", values=val_urb_string, default="?", type="predefined", color="(180,20,20)")
    ET.SubElement(relations, "Rel", name="AreaAgr", arity="1", argtypes="hru_agr", valtype="numeric", default="0.0", type="predefined", color="(180,20,20)")
    ET.SubElement(relations, "Rel", name="AreaUrb", arity="1", argtypes="hru_urb", valtype="numeric", default="0.0", type="predefined", color="(180,20,20)")

    ET.SubElement(relations, "Rel", name="SubType", arity="1", argtypes="sub", valtype="categorical", values="RES,SUB", default="SUB", type="predefined", color="(180,20,20)")
    ET.SubElement(relations, "Rel", name="Pollution", arity="1", argtypes="sub", valtype="categorical", values="LOW,MED,HIG", default="?", type="probabilistic", color="(180,20,20)")  

    ET.SubElement(relations, "Rel", name="target", arity="0", argtypes="sub", valtype="boolean", default="?", type="probabilistic", color="(180,20,20)")
    ET.SubElement(relations, "Rel", name="constr", arity="0", argtypes="sub", valtype="boolean", default="?", type="probabilistic", color="(180,20,20)")
    
    data_xml = ET.SubElement(root, "Data")
    data_for_input_domain = ET.SubElement(data_xml, "DataForInputDomain")
    domain = ET.SubElement(data_for_input_domain, "Domain")
    pos = nx.spring_layout(G, seed=888, iterations=100, scale=800)
    
    main_nodes = list(range(0, data['sub'].x.shape[0]))
    sub_graph = G.subgraph(main_nodes)
    pos_sub = nx.spring_layout(sub_graph, seed=888, scale=800, iterations=100)

    # write each node in the domain and assign a coordinate (for Bavaria)
    for i, (node, (x, y)) in enumerate(pos.items()):
        if node in main_nodes:
            min_x = min(coord[0] for coord in pos_sub.values())
            min_y = min(coord[1] for coord in pos_sub.values())
        else:
            min_x = min(coord[0] for coord in pos.values())
            min_y = min(coord[1] for coord in pos.values())

        ET.SubElement(domain, "obj", ind=str(i), name=str(i), coords="{x:.4f}".format(x=(x - min_x if min_x < 0 else x)+100)+","+"{y:.4f}".format(y=(y - min_y if min_y < 0 else y)+100))

    predefined_rels = ET.SubElement(data_for_input_domain, "PredefinedRels")

    # write all the nodes types 
    index_node = 0
    node_name = dict()
    for type in data.node_types:
        node_string = ""
        if type == 'sub':
            node_name[type] = []
            for node in range(data[type].x.size(0)):
                node_string += f"({node})"
                node_name[type].append(node)
                index_node += 1
            ET.SubElement(predefined_rels, "d", rel="sub", args=node_string, val="true")
        if type == 'hru_agr':
            node_name[type] = []
            for node in range(data[type].x.size(0)):
                node_string += f"({index_node})"
                node_name[type].append(index_node)
                index_node += 1
            ET.SubElement(predefined_rels, "d", rel="hru_agr", args=node_string, val="true")
        if type == 'hru_urb':
            node_name[type] = []
            for node in range(data[type].x.size(0)):
                node_string += f"({index_node})"
                node_name[type].append(index_node)
                index_node += 1
            ET.SubElement(predefined_rels, "d", rel="hru_urb", args=node_string, val="true")

    index_node = 0
    for type in data.edge_types:
        edge_string = ""
        if type == ('sub','to','sub'):
            for edges in data[type].edge_index.t():
                edge_string += f"({edges[0].item()},{edges[1].item()})"
                index_node += 1
            ET.SubElement(predefined_rels, "d", rel="sub_to_sub", args=edge_string, val="true")

        if type == ('hru_agr','to','sub'):
            for edges in data[type].edge_index.t():
                edge_string += f"({node_name['hru_agr'][edges[0]]},{edges[1].item()})"
                index_node += 1

            ET.SubElement(predefined_rels, "d", rel="hru_agr_to_sub", args=edge_string, val="true")

        if type == ('sub','to','hru_agr'):
            for edges in data[type].edge_index.t():
                edge_string += f"({edges[0].item()},{node_name['hru_agr'][edges[1]]})"
                index_node += 1
            ET.SubElement(predefined_rels, "d", rel="sub_to_hru_agr", args=edge_string, val="true")

        if type == ('hru_urb','to','sub'):
            for edges in data[type].edge_index.t():
                edge_string += f"({node_name['hru_urb'][edges[0]]},{edges[1].item()})"
                index_node += 1
            ET.SubElement(predefined_rels, "d", rel="hru_urb_to_sub", args=edge_string, val="true")

        if type == ('sub','to','hru_urb'):
            for edges in data[type].edge_index.t():
                edge_string += f"({edges[0].item()},{node_name['hru_urb'][edges[1]]})"
                index_node += 1
            ET.SubElement(predefined_rels, "d", rel="sub_to_hru_urb", args=edge_string, val="true")
    
    features_sub = data['sub'].x
    features_hru_agr = data['hru_agr'].x
    features_hru_urb = data['hru_urb'].x
    
    # write the subbasin nodes, in this case this type has a size of two (SUB, RES)!
    sub_string = ""
    res_string = ""
    for i, feature in enumerate(features_sub):
        assert feature.size(0) == 2, "Subbasin feature length not 2!"
        if feature[0] == 0:
            sub_string += f"({i})"
        else:
            res_string += f"({i})"

    ET.SubElement(predefined_rels, "d", rel="SubType", args=sub_string, val="SUB")
    ET.SubElement(predefined_rels, "d", rel="SubType", args=res_string, val="RES")
    
    # write the numberic features for agr and urb (area)
    for i, feature in enumerate(features_hru_agr):
        ET.SubElement(predefined_rels, "d", rel="AreaAgr", args=f"({node_name['hru_agr'][i]})", val=str(feature[-1].item()))
    
    for i, feature in enumerate(features_hru_urb):
        ET.SubElement(predefined_rels, "d", rel="AreaUrb", args=f"({node_name['hru_urb'][i]})", val=str(feature[-1].item()))
    
    # write for each urb node its corresponding value
    features_hru_urb =  all_data[0][0]['hru_urb'].x
    urb_dict = dict.fromkeys(val_urb, "")

    for i, feature in enumerate(features_hru_urb):
        val = np.argmax(feature[:len(val_urb)].numpy())
        urb_dict[val_urb[val]] += f"({node_name['hru_urb'][i]})"
    for key, value in urb_dict.items():
        if value != "":
            ET.SubElement(predefined_rels, "d", rel="LandUseUrb", args=value, val=key)

    if with_labels:
        for i in range(len(all_data)):
            data = all_data[i][0]
            features_hru_agr = data['hru_agr'].x

            # Create a new <probabilistic_rels_case> element for each iteration
            probabilistic_rels_case_instance = ET.SubElement(data_for_input_domain, "ProbabilisticRelsCase")

            ET.SubElement(probabilistic_rels_case_instance, "d", rel="target", args="()", val="true")
            ET.SubElement(probabilistic_rels_case_instance, "d", rel="constr", args="()", val="true")

            # Assign the land use value to a dictionary
            luse_dict = dict.fromkeys(val_luse, "")
            # urb_dict = dict.fromkeys(val_urb, "")
            
            for j, feature in enumerate(features_hru_agr):
                val = np.argmax(feature[:len(val_luse)].numpy())
                luse_dict[val_luse[val]] += f"({node_name['hru_agr'][j]})"
            
            for key, value in luse_dict.items():
                if value != "":
                    ET.SubElement(probabilistic_rels_case_instance, "d", rel="LandUse", args=value, val=key)

            # Assign the pollution value to a dictionary
            data_pollution = data['sub'].y
            pollution_dict = dict.fromkeys(val_pollution, "")
            
            for j, val in enumerate(data_pollution):
                pollution_dict[val_pollution[val.item()]] += f"({j})"
            
            for key, value in pollution_dict.items():
                if value != "":
                    ET.SubElement(probabilistic_rels_case_instance, "d", rel="Pollution", args=value, val=key)
    else:
        probabilistic_rels_case_instance = ET.SubElement(data_for_input_domain, "ProbabilisticRelsCase")
        ET.SubElement(probabilistic_rels_case_instance, "d", rel="target", args="()", val="true")
        ET.SubElement(probabilistic_rels_case_instance, "d", rel="constr", args="()", val="true")
        
    # write the file 
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(file_name, encoding='utf-8')

if __name__ == "__main__":
    data = pickle.load(open("data/test_graphs_all.pkl", "rb"))
    create_rdef_file(data, False, "water-network.rdef")