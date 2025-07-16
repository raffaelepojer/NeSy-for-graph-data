from .rbn_utils import *

def write_rbn_ACR_graph(rbn_path, model, prob_name, feature_names, feature_probs,
                         constraints=True, soft_prob=0.99, read_type="add", agg_type="add"):
    
    mlp = has_module(model, "mlp")
    gnn_layers = count_ACR_layers(model)

    rbn = write_prob_form(feature_names, feature_probs, include_edge=0.5)
    rbn += write_layer_zero(prob_name, feature_names)

    if constraints:
        rbn += write_constraints(feature_names, soft_prob=soft_prob)

    for layer in range(gnn_layers):
        if layer == 0:
            dim = count_dim_ACR_input_layer(model)
            rbn += write_read_formula(prob_name, layer, dim)
            rbn += write_agg_formula(prob_name, layer, dim)
        elif layer > 0 and mlp:
            dim = count_dim_ACR_layer(model, layer-1)
            rbn += write_read_formula(prob_name, layer, dim, count_mlp_layers(model, layer-1)-1)
            rbn += write_agg_formula(prob_name, layer, dim, count_mlp_layers(model, layer-1)-1)
        else:
            dim = count_dim_ACR_layer(model, layer-1)
            rbn += write_read_formula(prob_name, layer, dim)
            rbn += write_agg_formula(prob_name, layer, dim)

        rbn += write_rbn_layer_ACR(prob_name, layer, model, mlp=mlp)
    
    # final readout
    dim = count_dim_ACR_layer(model, gnn_layers-1)
    if mlp:
        rbn += write_read_formula(prob_name, count_ACR_layers(model), dim, count_mlp_layers(model, gnn_layers-1)-1, read_type=read_type)
    else:
        rbn += write_read_formula(prob_name, count_ACR_layers(model), dim, -1, read_type=read_type)

    # final classifier, two classes
    rbn += write_final_graph_classifier_ACR(prob_name, model)

    for i in range(2):
        rbn += negative_out(prob_name, i)
    for i in range(2):
        rbn += soft_max_out(prob_name, i)

    with open(rbn_path, "w") as f:
        f.writelines(rbn)
    f.close()

def write_rbn_ACR_node(rbn_path, model, prob_name, feature_names, feature_probs,
                         constraints=False, soft_prob=0.99, include_edge=0.0, categorical=[], read_type="add", agg_type="add", write_param=False):
    
    mlp = has_module(model, "mlp")
    gnn_layers = count_ACR_layers(model)

    rbn = write_prob_form(feature_names, feature_probs, categorical=categorical, include_edge=include_edge)
    rbn += write_layer_zero(prob_name, feature_names, categorical=categorical)

    if constraints:
        rbn += write_constraints(feature_names, soft_prob=soft_prob)

    for layer in range(gnn_layers):
        if layer == 0:
            dim = count_dim_ACR_input_layer(model)
            rbn += write_read_formula(prob_name, layer, dim)
            rbn += write_agg_formula(prob_name, layer, dim)
        elif layer > 0 and mlp:
            dim = count_dim_ACR_layer(model, layer-1)
            rbn += write_read_formula(prob_name, layer, dim, count_mlp_layers(model, layer-1)-1)
            rbn += write_agg_formula(prob_name, layer, dim, count_mlp_layers(model, layer-1)-1)
        else:
            dim = count_dim_ACR_layer(model, layer-1)
            rbn += write_read_formula(prob_name, layer, dim)
            rbn += write_agg_formula(prob_name, layer, dim)

        rbn += write_rbn_layer_ACR(prob_name, layer, model, mlp=mlp, write_param=write_param)
    
    # rbn output
    rbn += write_final_node_formula_ACR(prob_name, model, mlp=mlp, write_param=write_param)

    with open(rbn_path, "w") as f:
        f.writelines(rbn)
    f.close()