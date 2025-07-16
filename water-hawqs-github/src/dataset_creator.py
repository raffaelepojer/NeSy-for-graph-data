import networkx as nx
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from torch_geometric.data import HeteroData
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from plotting import *
import os
import pickle

#############################################################################################
## This file generate the graph given the HAWQS data and save them as pickle for later use ##
#############################################################################################

# node attributes for the subbasin nodes
def read_subb_feat(file_path, G, subbasin_data):
    def convert_float(value):
        try:
            return float(value)
        except ValueError:
            assert False
    node_type = nx.get_node_attributes(G, 'node_type')

    subbasins_feature = []
    for k in subbasin_data:
        subbasin_feature = []
        extensions = ['.pnd', '.rte']
        subbasin_feature.append(1.0 if node_type[k] == 'subbasin' else 0.0)
        for ex in extensions:
            dot_path = file_path + "/TxtInOut/" + subbasin_data[k].split('.')[0] + ex
            with open(dot_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if "|" in line:
                        value, rest = line.split("|")
                        value = value.strip()
                        if ex != '.sub':
                            subbasin_feature.append(convert_float(value))                            
                        elif ex == '.sub':
                            # add only this features for .sub
                            if 'SUB_ELEV' in rest or 'SUB_KM' in rest: 
                                subbasin_feature.append(convert_float(value))

        subbasins_feature.append(subbasin_feature)
        df = pd.DataFrame(subbasins_feature)
    return df

# node attributes for the HRU nodes
def add_hru_nodes(file_path, G):
    info_file = file_path + "/TxtInOut/input.std"

    with open(info_file, 'r') as file:
        lines = file.readlines()
        # look for the summary table
        hru_count = len(G.nodes())
        hrus_fetatures = []
        columns = []
        for line_idx, line in enumerate(lines):
            if "HRU Input Summary Table 1:" in line:
                start_data_idx = line_idx + 2
                for line in lines[start_data_idx:]:
                    hru_feature = []
                    if line.strip() == "":
                        break
                    data = line.split()
                    # insert the node index for the hru (#node+hru_idx)
                    hru_feature.append(int(data[1])+hru_count)
                    for i, d in enumerate(data):
                        G.add_node(int(data[1])+hru_count, node_type='hru')
                        G.add_edge(int(data[1])+hru_count, int(data[0]), color='black', weight=0.1, arrowsize=3)    
                        G.add_edge(int(data[0]), int(data[1])+hru_count, color='black', weight=0.1, arrowsize=3)
                        if i > 1: # exclude the first two columns
                            hru_feature.append(float(data[i]))
                    hrus_fetatures.append(hru_feature)
            if "HRU CN Input Summary Table:" in line:
                start_data_idx = line_idx + 2
                for idx, line in enumerate(lines[start_data_idx:]):
                    hru_feature = []
                    if line.strip() == "":
                        break
                    data = line.split()
                    if len(data) > 11:
                        data[4:6] = [" ".join(data[4:6])]
                    for i, d in enumerate(data):
                        if i > 4: 
                            hru_feature.append(float(data[i]))
                        elif i == 3 or i == 4:
                            hru_feature.append(data[i])
                    hrus_fetatures[idx] += hru_feature
            if "HRU Input Summary Table 2:" in line:                
                start_data_idx = line_idx + 2
                for idx, line in enumerate(lines[start_data_idx:]):
                    hru_feature = []
                    if line.strip() == "":
                        break
                    data = line.split()
                    if len(data) > 12:
                        data[3:5] = [" ".join(data[3:5])]
                    for i, d in enumerate(data):
                        if i == 4: 
                            hru_feature.append(data[i])
                        elif i > 4:
                            hru_feature.append(float(data[i]))
                    hrus_fetatures[idx] += hru_feature
            if "HRU Input Summary Table 3:" in line:                
                start_data_idx = line_idx + 2
                for idx, line in enumerate(lines[start_data_idx:]):
                    hru_feature = []
                    if line.strip() == "":
                        break
                    data = line.split()
                    if len(data) == 4:
                        hru_feature.append(0.0)
                        hru_feature.append(float(data[-1]))
                    elif len(data) == 5:
                        hru_feature.append(1.0)
                        hru_feature.append(float(data[-1]))
                    hrus_fetatures[idx] += hru_feature
            if "HRU Input Summary Table 4 (Groundwater):" in line:                
                start_data_idx = line_idx + 2
                for idx, line in enumerate(lines[start_data_idx:]):
                    hru_feature = []
                    if line.strip() == "":
                        break
                    data = line.split()
                    for i, d in enumerate(data):
                        if i > 2:
                            hru_feature.append(float(data[i]))
                    hrus_fetatures[idx] += hru_feature
        
        df = pd.DataFrame(hrus_fetatures)
    return G, df

# generate the graph structure of the watershed according to the specifications
def create_watershed_graph(file_path):
    def trace_source(storage):
        if storage in add_operations:
            input_1, input_2 = add_operations[storage]
            return trace_source(input_1) + trace_source(input_2)
        elif storage in hydrograph_storage:
            return [(hydrograph_storage[storage], 'subbasin')]
        elif storage in routres_operations:
            return [(routres_operations[storage], 'reservoir')]
        return []
    
    G = nx.DiGraph()
    hydrograph_storage = {}
    add_operations = {}
    routres_operations = {}
    subbasin_data = {}

    subbasins = set()

    subbasin_file = file_path + "/TxtInOut/fig.fig"

    with open(subbasin_file, 'r') as file:
        lines = file.readlines()

    for idx_line, line in enumerate(lines):
        parts = line.split()

        if line.startswith("subbasin"):
            storage_location = parts[2]  # Hydrograph storage location (column 2)
            subbasin_id = parts[3]  # Subbasin number (last column)
            hydrograph_storage[storage_location] = subbasin_id

            subbasins.add(subbasin_id)
            G.add_node(int(subbasin_id), label=subbasin_id)
            subbasin_data[int(subbasin_id)] = lines[idx_line+1].strip()
        
        elif line.startswith("add"):
            result_storage = parts[2]  # Where the combined loadings are stored (column 2)
            input_storage_1 = parts[3]  # First hydrograph storage (column 3)
            input_storage_2 = parts[4]  # Second hydrograph storage (column 4)

            add_operations[result_storage] = (input_storage_1, input_storage_2)
    
        elif line.startswith("route"):
            result_storage = parts[2]  # Where the result of the route is stored (column 2)
            target_subbasin = parts[3]  # Reach number (target subbasin) (column 3)
            input_storage = parts[4]  # Hydrograph storage from the previous step (column 4)

            sources = trace_source(input_storage)

            for source, sub_type in sources:
                # avoid self loops
                if source != target_subbasin:
                    G.add_node(int(source), node_type=sub_type)
                    G.add_node(int(target_subbasin), node_type=sub_type)
                    G.add_edge(int(source), int(target_subbasin), color='blue', weight=0.8, arrowsize=10)

            hydrograph_storage[result_storage] = target_subbasin
        
        elif line.startswith("routres"):
            result_storage = parts[2]  # Outflow from the reservoir is stored here
            reservoir_number = parts[3]  # Reservoir number
            input_storage = parts[4]  # Hydrograph storage location where input to reservoir is located
            sources = trace_source(input_storage)
            for source, sub_type in sources:
                routres_operations[result_storage] = source
    
    return G, subbasin_data

def parse_watershed_file(file_path):
    G_sub, subbasin_data = create_watershed_graph(file_path)
    sub_df = read_subb_feat(file_path, G_sub, subbasin_data)  
    G, hru_df = add_hru_nodes(file_path, G_sub)
    return G, sub_df, hru_df

# import y
def read_out_reach(file_path):
    return pd.read_csv(file_path + '/TxtInOut/output-rch.csv', delimiter=',')

def to_categorical(data, classes):
    cat_array = np.zeros((data.shape[0], len(classes)))
    for idx, _ in enumerate(cat_array):
        cat_array[idx, classes.index(data[idx])] = 1.0
    return cat_array

def get_yearly_avg(df, parameters):
    # Returns the yearly average of the specified parameter(s) for each subbasin node.
    
    if isinstance(parameters, str):
        parameters = [parameters]
    df_yearly_avg = df.groupby(['Year', 'RCH'])[parameters].mean().reset_index()
    return df_yearly_avg


def create_hetero_data_y_m(year, month, G, y_cat, sub_data, hru_data, reach_data, parameter, set_luse, set_other, scenario=None):
    # convert the data to a heterogenous graph
    # in this case many features for the nodes are not included for simplicity
    
    # Convert dataframe string to one-hot encoding
    # it also fix the columns index
    numeric_df_hru_agr = hru_data.copy()
    numeric_df_hru_non_arg = hru_data.copy()

    # extract all the rows that are not in the agr list
    numeric_df_hru_agr = numeric_df_hru_agr[numeric_df_hru_agr[9].isin(set_luse)]
    numeric_df_hru_non_arg = numeric_df_hru_non_arg[~numeric_df_hru_non_arg[9].isin(set_luse)]

    # take the values of the columns and remove them from the dataframe
    luse_agr = numeric_df_hru_agr.pop(9)
    soil_agr = numeric_df_hru_agr.pop(10)
    bo_agr = numeric_df_hru_agr.pop(17)
    luse_non_agr = numeric_df_hru_non_arg.pop(9)
    soil_non_agr = numeric_df_hru_non_arg.pop(10)
    others_non_agr = numeric_df_hru_non_arg.pop(17)
    # insert the values in the first columns with the same column name
    numeric_df_hru_agr.insert(1, 9, luse_agr)
    numeric_df_hru_agr.insert(2, 10, soil_agr)
    numeric_df_hru_agr.insert(3, 17, bo_agr)
    numeric_df_hru_non_arg.insert(1, 9, luse_non_agr)
    numeric_df_hru_non_arg.insert(2, 10, soil_non_agr)
    numeric_df_hru_non_arg.insert(3, 17, others_non_agr)
    # reset the columns index
    numeric_df_hru_agr.columns = range(numeric_df_hru_agr.columns.size)
    numeric_df_hru_non_arg.columns = range(numeric_df_hru_non_arg.columns.size)
    # convert the values to numpy array
    one_hot_luse_agr = numeric_df_hru_agr.iloc[:, 1].to_numpy()
    one_hot_luse_non_agr = numeric_df_hru_non_arg.iloc[:, 1].to_numpy()
    # convert the values to one-hot encoding
    one_hot_luse_agr = to_categorical(one_hot_luse_agr, set_luse)
    one_hot_luse_non_agr = to_categorical(one_hot_luse_non_agr, set_other)
    # take the hru number
    hru_number_arg = numeric_df_hru_agr.iloc[:, 0]
    hru_number_non_arg = numeric_df_hru_non_arg.iloc[:, 0]
    # remove the columns that are not needed
    numeric_df_hru_agr.drop(numeric_df_hru_agr.columns[0:4], axis=1, inplace=True)
    numeric_df_hru_non_arg.drop(numeric_df_hru_non_arg.columns[0:4], axis=1, inplace=True)
    # scale the values (area) (second column of the hru_data)
    scaler = MinMaxScaler()
    scaled_area = scaler.fit_transform(hru_data.iloc[:, [1]])
    area_df_hru_arg = pd.DataFrame(scaled_area[:len(numeric_df_hru_agr)])
    area_df_hru_non_arg = pd.DataFrame(scaled_area[len(numeric_df_hru_agr):])

    # scaler = MinMaxScaler()
    # numeric_df_hru_agr = pd.DataFrame(scaler.fit_transform(numeric_df_hru_agr), columns=numeric_df_hru_agr.columns)
    # numeric_df_hru_non_arg = pd.DataFrame(scaler.fit_transform(numeric_df_hru_non_arg), columns=numeric_df_hru_non_arg.columns)
    # take the area column
    # area_df_hru_arg = numeric_df_hru_agr.iloc[:, 0]
    # area_df_hru_non_arg = numeric_df_hru_non_arg.iloc[:, 0]
    
    # concatenate the one-hot encoding with the area column, include also the first column
    numeric_df_hru_agr = pd.concat([pd.DataFrame(one_hot_luse_agr), area_df_hru_arg], axis=1)
    numeric_df_hru_non_arg = pd.concat([pd.DataFrame(one_hot_luse_non_agr), area_df_hru_non_arg], axis=1)
    # reset the columns index
    numeric_df_hru_agr.columns = range(numeric_df_hru_agr.columns.size)
    numeric_df_hru_non_arg.columns = range(numeric_df_hru_non_arg.columns.size)
    
    numeric_df_sub = sub_data.copy()
    reservoir = numeric_df_sub.pop(0)
    numeric_df_sub = pd.DataFrame(scaler.fit_transform(numeric_df_sub), columns=numeric_df_sub.columns)
    one_hot_res = pd.get_dummies(reservoir)
    
    # If reach data is provided and month is None, include yearly simulation data
    if reach_data is not None and month is None:
        df_yearly = get_yearly_avg(reach_data, ['FLOW_IN', 'FLOW_OUT', 'EVAP', 'TLOSS', 'SED_IN', 'SED_OUT', 'SEDCONC'])
        if year is not None:
            df_yearly = df_yearly[df_yearly['Year'] == year]
        else:
            # Compute overall average across all years:
            df_yearly = df_yearly.drop(columns=['Year', 'RCH']).mean(axis=0).to_frame().T
        scaled_features = scaler.fit_transform(df_yearly)
        scaled_df = pd.DataFrame(scaled_features, columns=df_yearly.columns)
        numeric_df_sub = pd.concat([one_hot_res, scaled_df], axis=1)
    else:
        # Otherwise, include only the fixed features from the sub data
        numeric_df_sub = pd.DataFrame(one_hot_res)

    numeric_df_sub.columns = range(numeric_df_sub.columns.size)
    
    data = HeteroData()
    # delete all attributes of the nx graph
    sub_edges = []
    for edge in G.edges():
        if ((G.nodes[edge[0]]['node_type'] != 'hru') and (G.nodes[edge[1]]['node_type'] != 'hru')):
            sub_edges.append(edge)

    hru_edges = G.edges(list(range(24, G.number_of_nodes()+1)))
    
    data['sub'].x = torch.tensor(numeric_df_sub.to_numpy(dtype=float), dtype=torch.float)
    data['agr'].x = torch.tensor(numeric_df_hru_agr.to_numpy(dtype=float), dtype=torch.float)
    data['urb'].x = torch.tensor(numeric_df_hru_non_arg.to_numpy(dtype=float), dtype=torch.float)

    data['sub', 'downstream', 'sub'].edge_index = torch.tensor(
        np.array([(edge[0]-1, edge[1]-1) for edge in sub_edges]).T, dtype=torch.long)

    hru_agr_edges = []
    for edge in hru_edges:
        if edge[0] in hru_number_arg.to_numpy(dtype=int):
            hru_agr_edges.append(edge)
    # Build a dictionary to reindex the hru numbers starting from 0
    hru_number_arg_dict = {hru: idx for idx, hru in enumerate(hru_number_arg.to_numpy(dtype=int))}
    hru_agr_edges = torch.tensor(
        np.array([(hru_number_arg_dict[edge[0]], edge[1]-1) for edge in hru_agr_edges]).T, dtype=torch.long)

    hru_non_arg_edges = []
    for edge in hru_edges:
        if edge[0] in hru_number_non_arg.to_numpy(dtype=int):
            hru_non_arg_edges.append(edge)
    hru_number_non_arg_dict = {hru: idx for idx, hru in enumerate(hru_number_non_arg.to_numpy(dtype=int))}
    hru_non_arg_edges = torch.tensor(
        np.array([(hru_number_non_arg_dict[edge[0]], edge[1]-1) for edge in hru_non_arg_edges]).T, dtype=torch.long)

    data['agr', 'downstream_agr', 'sub'].edge_index = hru_agr_edges
    data['urb', 'downstream_urb', 'sub'].edge_index = hru_non_arg_edges

    data['sub', 'upstream_agr', 'agr'].edge_index = torch.empty((2, 0), dtype=torch.long)
    data['sub', 'upstream_urb', 'urb'].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Assign target labels. When month is provided, we use both year and month.
    # If year is None, we compute the average over all years.
    if month is not None:
        if year is not None:
            y_values = y_cat[(y_cat['Year'] == year) & (y_cat['Month'] == month)][parameter].to_numpy(dtype=float)
            data['sub'].y = torch.tensor(y_values, dtype=torch.long)
        else:
            # If no year is specified but a month is, average that month across years.
            avg_value = y_cat[y_cat['Month'] == month][parameter].mean()
            data['sub'].y = torch.tensor(np.array([avg_value] * data['sub'].x.shape[0]), dtype=torch.long)
    else:
        if year is not None:
            y_values = y_cat[y_cat['Year'] == year][parameter].to_numpy(dtype=float)
            data['sub'].y = torch.tensor(y_values, dtype=torch.long)
        else:
            y_values = y_cat[parameter].to_numpy(dtype=float)
            data['sub'].y = torch.tensor(y_values, dtype=torch.long)
    
    if scenario is not None:
        data.scenario = scenario

    return data

def make_equal_width_bins(min_value, max_value, values, num_bins):
    bins = np.linspace(min_value, max_value, num_bins + 1)
    new_bins = []
    for v in values:
        if v > max_value:
            new_bins.append(num_bins - 1)  
        else:
            for idx, b in enumerate(bins):
                if v <= b:
                    new_bins.append(idx - 1)
                    break
    return new_bins

def make_logarithmic_bins(min_value, max_value, values, num_bins):
    bins = np.logspace(np.log10(min_value + 1), np.log10(max_value + 1), num_bins + 1)
    new_bins = []
    for v in values:  
        v = v + 1
        if v > max_value:
            new_bins.append(num_bins - 1)  
        else:
            for idx, b in enumerate(bins):
                if v <= b:
                    new_bins.append(idx - 1)
                    break
    return new_bins

def make_equal_frequency_bins(values, num_bins):
    return pd.qcut(values.values, q=num_bins, labels=False)

def cat_df_values_yearly1(df, parameter, binning_type, max_value, num_bins=5):
    y_cat = df.copy()
    y_cat_mean = y_cat.groupby(['scenario', 'Year', 'RCH'], group_keys=True).mean()
    y_cat_max = y_cat.groupby(['scenario', 'Year', 'RCH'], group_keys=True).max()
    y_cat_mean = y_cat_mean.reset_index()
    y_cat_max = y_cat_max.reset_index()
    
    y_cat_mean = y_cat_mean[['scenario', 'RCH', 'Year', parameter]]
    y_cat_max = y_cat_max[['scenario', 'RCH', 'Year', parameter]]

    real_nvalues_mean = y_cat_mean[parameter]
    real_nvalues_max = y_cat_max[parameter]
    
    if binning_type == 'equ':
        cat_nvalues_mean = make_equal_width_bins(0, max_value, real_nvalues_mean, num_bins)
        cat_nvalues_max = make_equal_width_bins(0, max_value, real_nvalues_max, num_bins)
    elif binning_type == 'freq':
        cat_nvalues_mean = make_equal_frequency_bins(real_nvalues_mean, num_bins)
        cat_nvalues_max = make_equal_frequency_bins(real_nvalues_max, num_bins)
    elif binning_type == 'log':
        cat_nvalues_mean = make_logarithmic_bins(0, max_value, real_nvalues_mean, num_bins)
        cat_nvalues_max = make_logarithmic_bins(0, max_value, real_nvalues_max, num_bins)

    y_cat_mean[parameter] = cat_nvalues_mean
    y_cat_max[parameter] = cat_nvalues_max
    y_cat_mean[parameter] = y_cat_mean[parameter].astype(float)
    y_cat_max[parameter] = y_cat_max[parameter].astype(float)
    return y_cat_mean, y_cat_max

def average_by_year_y(df, parameter, num_bins=3):
    y_cat = df.copy()
    y_cat_mean = y_cat.groupby(['scenario', 'RCH'], group_keys=True).mean()
    y_cat_mean = y_cat_mean.reset_index()
    y_cat_mean = y_cat_mean[['scenario', 'RCH', parameter]]
    real_nvalues_mean = y_cat_mean[parameter]
    cat_nvalues_mean = make_equal_frequency_bins(real_nvalues_mean, num_bins)
    y_cat_mean[parameter] = cat_nvalues_mean
    y_cat_mean[parameter] = y_cat_mean[parameter].astype(float)
    return y_cat_mean


####################################################################################################
####################################################################################################
####################################################################################################

def from_swat_to_torch(base_path: str,
                       scenarios: list,
                       num_bins: int,
                       agr_types: list,
                       y_param: str,
                       all_years: bool = True,
                       use_rech: bool = False
                       ):

    set_luse = set();set_other = set();set_soil = set();set_bo = set()
    pyg_data = []

    total_df = pd.DataFrame()
    for scenario in scenarios:
        file_path = base_path + scenario
        G, sub_data, hru_data = parse_watershed_file(file_path)
        set_luse.update(hru_data[9].unique())
        set_other.update(hru_data[9].unique())
        set_soil.update(hru_data[10].unique())
        set_bo.update(hru_data[17].unique())
        reach_df = read_out_reach(file_path)
        reach_df.insert(0, 'scenario', scenario)
        reach_df['scenario'] = scenario
        total_df = pd.concat([total_df, reach_df], axis=0)

    # remove arg from set_other
    set_other = set_other - set(agr_types)
    # remove everything that is not agr from set_luse
    set_luse = set_luse.intersection(set(agr_types))

    luse = sorted(list(set_luse));other = sorted(list(set_other));soil = sorted(list(set_soil));bo = sorted(list(set_bo))

    if all_years:
        total_df = total_df.reset_index(drop=True)
        y_n_year_mean, y_n_year_max  = cat_df_values_yearly1(total_df, y_param, 'freq', 5, num_bins)
        y_n = y_n_year_mean.copy()
        years = list(range(2010, 2021))
        for year in years:
            for scenario in scenarios:
                G, sub_data, hru_data = parse_watershed_file(base_path + scenario)
                df_y =  y_n[(y_n['Year'] == year) & (y_n['scenario'] == scenario)]
                reach_df = read_out_reach(base_path + scenario)
                if use_rech:
                    data = create_hetero_data_y_m(year, None, G, df_y, sub_data, hru_data, reach_df, y_param, luse, other, scenario)
                else:
                    data = create_hetero_data_y_m(year, None, G, df_y, sub_data, hru_data, None, y_param, luse, other, scenario)
                pyg_data.append((data, G, year, -1))        
    else:
        y_n_year_mean  = average_by_year_y(total_df, y_param, num_bins)
        for scenario in scenarios:
            G, sub_data, hru_data = parse_watershed_file(base_path + scenario)
            df_y = y_n_year_mean[(y_n_year_mean['scenario'] == scenario)]
            data = create_hetero_data_y_m(None, None, G, df_y, sub_data, hru_data, None, y_param, luse, other, scenario)
            pyg_data.append((data, G, -1, -1))

    # First split: 70% train+val, 30% test
    stratify = None
    if pyg_data[0][2] != -1:
        stratify = [graph[0].scenario for graph in pyg_data]

    train_val_graphs, test_graphs = train_test_split(
        pyg_data,
        test_size=0.3,
        stratify=stratify,
        random_state=42
    )

    stratify = None
    if pyg_data[0][2] != -1:
        stratify = [graph[0].scenario for graph in train_val_graphs]

    train_graphs, val_graphs = train_test_split(
        train_val_graphs,
        test_size=0.2143,  # 15/70 of overall data
        stratify=stratify,
        random_state=42
    )
    return train_graphs, val_graphs, test_graphs


def main():     
    current_dir = os.path.dirname(os.path.abspath(__file__))

    base_path = os.path.join(current_dir, "..", "scenarios") + os.sep
    # scenarios = ["default-2", "soy-3", "corn-3", "pasture-3", 'cosy-3', "pasture-corn", "soy-corn-2", "cosy-corn-2", "pasture-soy-2", "pasture-cosy-2", "cosy-soy-2", "default-mix", "default-mix-2", "default-mix-3"]
    scenarios = ["default-2"]
    agr_types = ['CORN', 'COSY', 'PAST', 'SOYB']

    all_years = True
    train_graphs, val_graphs, test_graphs = from_swat_to_torch(
        base_path=base_path,
        scenarios=scenarios,
        num_bins=3,
        agr_types=agr_types,
        y_param='TOT_N_CONC',
        all_years=all_years
    )

    plot_all_graphs(scenarios, (train_graphs + val_graphs + test_graphs), 3, all_years=all_years)

    data_dir = os.path.join(current_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'train_graphs_all.pkl'), 'wb') as f:
        pickle.dump(train_graphs, f)
        print(len(train_graphs))
    with open(os.path.join(data_dir, 'test_graphs_all.pkl'), 'wb') as f:
        pickle.dump(test_graphs, f)
        print(len(test_graphs))
    with open(os.path.join(data_dir, 'val_graphs_all.pkl'), 'wb') as f:
        pickle.dump(val_graphs, f)
        print(len(val_graphs))

if __name__ == '__main__':
    main()