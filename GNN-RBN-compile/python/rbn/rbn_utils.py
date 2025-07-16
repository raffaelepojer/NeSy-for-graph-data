import torch
import re

def read_model(model_path):
    model = torch.load(model_path)
    model_dict = model.state_dict()
    print(model_dict)
    return model_dict

def load_model(model):
    print(model.state_dict())

def count_ACR_layers(model):
  """Counts the number of layers in a ACR model."""
  return len(model.layers)

def count_mlp_layers(model, layer):
    """Counts the number of mlp layers in a specific ACR layer."""
    return len(model.layers[layer].mlp.linears)

def count_dim_ACR_layer(model, layer):
    """Counts the dimension of each layer in a ACR model."""
    return model.layers[layer].A.weight.shape[0]

def count_dim_ACR_input_layer(model):
    """Counts the dimension of the input ACR model."""
    return model.layers[0].A.weight.shape[1]

def extract_weights(model, layer_name):
  """Extracts the weights of a PyTorch model from layers whose names start with a specific prefix."""
  weights = []
  for name, layer in model.named_parameters():
    if name == layer_name:
      weights.extend(layer.data)
  return weights

def extract_weights_prefix(model, prefix):
  """Extracts the weights of a PyTorch model from layers whose names start with a specific prefix."""
  weights = []
  for name, layer in model.named_parameters():
    if name.startswith(prefix):
      weights.extend(layer.data)
  return weights

def has_module(model, string):
  """Checks if a PyTorch model has a module with a name containing a specific string."""
  for name, module in model.named_modules():
    if re.search(string, name):
      return True
  return False

def write_prob_form(feature_names, feature_prob, categorical=[], include_edge=0.5):
    """Writes the initial form of the RBN.
    Args:   feature_names: list of strings
            feature_prob: list of floats
            include_edge: float (default 0.5) if higher than 0, the edge is included in the RBN definition
    Returns: string 
    """
    result = ""
    if len(categorical) > 0:
        for i in range(len(feature_names)):
            result += '{}([node]v) = SOFTMAX '.format(feature_names[i])
            for j in range(len(categorical)-1):
                result += '{},'.format(feature_prob[i])
            result += '{};\n'.format(feature_prob[i])
    else:
        for i in range(len(feature_names)):
            result += '{}([node]v) = {};\n'.format(feature_names[i], feature_prob[i])
    if include_edge > 0:
        result += 'edge(v,w) = {};\n'.format(include_edge)

    result += '\n'
    return result

def write_layer_zero(prob_name, feature_names, categorical=[]):
    result = ""
    if len(categorical) > 0:
        for j, n_feature in enumerate(feature_names):
            for i, n_val in enumerate(categorical):        
                result += '@{}_layer_0_{}([node]v) = {}(v)={};\n'.format(prob_name, i, n_feature, n_val)
    else:
        for i, n_feature in enumerate(feature_names):
            # TODO add generative rule
            result += '@{}_layer_0_{}([node]v) = {}(v);\n'.format(prob_name, i, n_feature)
    result += '\n'
    return result

def write_constraints(feature_names, soft_prob):
    """Writes the all the possible combination of constraints to the RBN using soft probability.
    Args:   feature_names: list of strings
            soft_prob: float
    Returns: string
    """
    result = ""
    for i, const1 in enumerate(feature_names):
        for const2 in range(i, len(feature_names)):
            if const1 != feature_names[const2]:
                result += "const_"+const1+"_"+feature_names[const2] + \
                    "(v) = ((" + const1 + "(v) & " + feature_names[const2] + \
                    "(v)) * " + str(soft_prob) + ");\n"

    result += "all_const(v) = (("
    for i, const1 in enumerate(feature_names):
        if i < len(feature_names)-1:
            result += const1 + "(v) | "
        else:
            result += const1 + "(v)) * " + str(soft_prob) + ");\n"
    result += '\n'
    return result

def write_read_formula(prob_name, current_layer, current_dim, mlp_layer=-1, read_type="add"):
    """Writes the read formula for a specific layer of the RBN.
    Args:   prob_name: string
            current_layer: int 
            current_dim: int
            mlp_layer: int (default -1) if higher than 0, the input refers to the last mlp layer
            read_type: string (default "add") can be "add" or "mean"
    Returns: string
    """
    result = ""
    for idx in range(current_dim):
        result += "@" + prob_name + "_read_" + \
            str(current_layer) + "_" + str(idx) + "() = "
        result += "COMBINE "

        if mlp_layer >= 0 and current_layer > 0:
            result += "@" + prob_name + "_mlp_" + \
                str(current_layer) + "_" + str(mlp_layer) + "_" + str(idx) + "(w)\n"
        else:
            result += "@" + prob_name + "_layer_" + \
                    str(current_layer) + "_" + str(idx) + "(w)\n"
        
        if read_type == "add":
            result += "\tWITH sum\n\tFORALL w\n\tWHERE "
        elif read_type == "mean":
            result += "\tWITH mean\n\tFORALL w\n\tWHERE "
        else:
            print("illegal argument {} for read_type".format(read_type))
        result += "node(w);\n"
    result += '\n'
    return result

def write_agg_formula(prob_name, current_layer, current_dim, mlp_layer=-1, agg_type="add"):
    """Writes the aggregation formula for a specific layer of the RBN.
    Args:   prob_name: string
            current_layer: int
            current_dim: int
            mlp_layer: int (default -1) if higher than 0, the input refers to the last mlp layer
            agg_type: string (default "add") can be "add" or "mean"
    Returns: string
    """
    result = ""
    for idx in range(current_dim):
        result += "@" + prob_name + "_agg_" + \
                str(current_layer) + "_" + str(idx) + "([node]v) = "
        result += "COMBINE "

        if mlp_layer >= 0 and current_layer > 0:
            result += "(@" + prob_name + "_mlp_" + \
                str(current_layer) + "_" + str(mlp_layer) + "_" + str(idx) + "(w) * (edge(v,w)|edge(w,v)))\n"
        else:
            result += "(@" + prob_name + "_layer_" + \
                str(current_layer) + "_" + str(idx) + "(w) * (edge(v,w)|edge(w,v)))\n"
            
        if agg_type == "add":
            result += "\tWITH sum\n\tFORALL w\n\tWHERE "
        elif agg_type == "mean":
            result += "\tWITH mean\n\tFORALL w\n\tWHERE "
        else:
            print("illegal argument {} for agg_type".format(agg_type))
        result += "true;\n"
    result += '\n'
    return result

def write_rbn_layer_ACR(prob_name, current_layer, model_weights, 
                        mlp=False, read_type="add", agg_type="add", write_param=False):
    # if write_param is set to True, write a Primula parameter ($param) instead of the learnth weights
    result = ""
    
    if write_param:
        # Use Primula parameters ($param) instead of learned weights
        # Get dimensions from weights for parameter naming
        A_weights = extract_weights(model_weights, "layers."+str(current_layer)+".A.weight")
        A_bias = extract_weights(model_weights, "layers."+str(current_layer)+".A.bias")
        C_weights = extract_weights(model_weights, "layers."+str(current_layer)+".C.weight")
        C_bias = extract_weights(model_weights, "layers."+str(current_layer)+".C.bias")
        R_weights = extract_weights(model_weights, "layers."+str(current_layer)+".R.weight")
        R_bias = extract_weights(model_weights, "layers."+str(current_layer)+".R.bias")

        for current_row in range(len(A_weights)):
            result += "@" + prob_name + "_layer_" + str(current_layer+1) + "_" + \
                str(current_row) + "([node]v) = COMBINE\n"
            
            for idx in range(len(C_weights[current_row])):
                result += "\t($C_{}_{}_{}*@" \
                    .format(current_layer, current_row, idx) + \
                    prob_name + "_layer_" + str(current_layer) + "_" + str(idx) + "(v)),\n"
            
            for idx in range(len(A_weights[current_row])):
                result += "\t($A_{}_{}_{}*@" \
                    .format(current_layer, current_row, idx) + \
                    prob_name + "_agg_" + str(current_layer) + "_" + str(idx) + "(v)),\n"
            
            for idx in range(len(R_weights[current_row])-1):
                result += "\t($R_{}_{}_{}*@" \
                    .format(current_layer, current_row, idx) + \
                    prob_name + "_read_" + str(current_layer) + "_" + str(idx) + "()),\n"
            
            idx = len(R_weights[current_row])-1
            if len(A_bias) > 0:
                result += "\t($R_{}_{}_{}*@" \
                    .format(current_layer, current_row, idx) + \
                    prob_name + "_read_" + str(current_layer) + "_" + str(idx) + "()),\n"
            else:
                result += "\t($R_{}_{}_{}*@" \
                    .format(current_layer, current_row, idx) + \
                    prob_name + "_read_" + str(current_layer) + "_" + str(idx) + "())\n"

            # write bias terms
            if len(A_bias) > 0:
                result += "\t$C_bias_{}_{}".format(current_layer, current_row) + ",\n"
                result += "\t$A_bias_{}_{}".format(current_layer, current_row) + ",\n"
                result += "\t$R_bias_{}_{}".format(current_layer, current_row) + "\n"

            if mlp:
                result += "\tWITH sum\n\tFORALL;\n"
            else:
                result += "\tWITH l-reg\n\tFORALL;\n"

            result += "\n"
        if mlp:
            result += write_mlp_layer_ACR(prob_name, current_layer, model_weights)
        return result
    
    A_weights = extract_weights(model_weights, "layers."+str(current_layer)+".A.weight")
    A_bias = extract_weights(model_weights, "layers."+str(current_layer)+".A.bias")
    C_weights = extract_weights(model_weights, "layers."+str(current_layer)+".C.weight")
    C_bias = extract_weights(model_weights, "layers."+str(current_layer)+".C.bias")
    R_weights = extract_weights(model_weights, "layers."+str(current_layer)+".R.weight")
    R_bias = extract_weights(model_weights, "layers."+str(current_layer)+".R.bias")

    for current_row in range(len(A_weights)):
        result += "@" + prob_name + "_layer_" + str(current_layer+1) + "_" + \
            str(current_row) + "([node]v) = COMBINE\n"
        
        for idx in range(len(C_weights[current_row])):
            result += "\t(" f"{C_weights[current_row][idx].item():.30f}" + "*@" + \
                prob_name + "_layer_" + str(current_layer) + "_" + str(idx) + "(v)),\n"
            
        for idx in range(len(A_weights[current_row])):
            result += "\t(" f"{A_weights[current_row][idx].item():.30f}" + "*@" + \
                prob_name + "_agg_" + str(current_layer) + "_" + str(idx) + "(v)),\n"
        
        # add a comma to the last line if there is the bias term to write
        for idx in range(len(R_weights[current_row])-1):
            result += "\t(" f"{R_weights[current_row][idx].item():.30f}" + "*@" + \
                prob_name + "_read_" + str(current_layer) + "_" + str(idx) + "()),\n"
        
        idx = len(R_weights[current_row])-1
        if len(A_bias) > 0:
            result += "\t(" f"{R_weights[current_row][-1].item():.30f}" + "*@" + \
                prob_name + "_read_" + str(current_layer) + "_" + str(idx) + "()),\n"
        else:
            result += "\t(" f"{R_weights[current_row][-1].item():.30f}" + "*@" + \
                prob_name + "_read_" + str(current_layer) + "_" + str(idx) + "())\n"

        # write bias terms
        if len(A_bias) > 0:
            result += "\t" + f"{C_bias[current_row].item():.30f}" + ",\n"
            result += "\t" + f"{A_bias[current_row].item():.30f}" + ",\n"
            result += "\t" + f"{R_bias[current_row].item():.30f}" + "\n"

        if mlp:
            result += "\tWITH sum\n\tFORALL;\n"
        else:
            result += "\tWITH l-reg\n\tFORALL;\n"

        result += "\n"
    
    if mlp:
        result += write_mlp_layer_ACR(prob_name, current_layer, model_weights)

    return result

def write_mlp_layer_ACR(prob_name, current_layer, model):
    result = ""
    for current_mlp_layer in range(count_mlp_layers(model, current_layer)):
        if has_module(model, "mlp.linears."):
            mlp_weights = extract_weights(model, "layers."+str(current_layer)+".mlp.linears."+str(current_mlp_layer)+".weight")
            mlp_bias = extract_weights(model, "layers."+str(current_layer)+".mlp.linears."+str(current_mlp_layer)+".bias")

        for current_mlp_row in range(len(mlp_weights)):
            # the fist layer takes the output of the RBN layer as input
            if current_mlp_layer == 0:
                result += "@" + prob_name + "_mlp_" + str(current_layer+1) + "_" + \
                        str(current_mlp_layer) + "_" + str(current_mlp_row) + "([node]v) = COMBINE\n"

                for idx in range(len(mlp_weights[current_mlp_row])):
                    result += "\t(" f"{mlp_weights[current_mlp_row][idx].item():.30f}" + "*@" + \
                        prob_name + "_layer_" + str(current_layer+1) + "_" + str(idx) + "(v)),\n"
                    
            # the other layers take the output of the previous layer as input
            elif current_mlp_layer > 0:
                result += "@" + prob_name + "_mlp_" + str(current_layer+1) + "_" + \
                        str(current_mlp_layer) + "_" + str(current_mlp_row) + "([node]v) = COMBINE\n"

                for idx in range(len(mlp_weights[current_mlp_row])):
                    result += "\t(" f"{mlp_weights[current_mlp_row][idx].item():.30f}" + "*@" + \
                        prob_name + "_mlp_" + str(current_layer+1) + "_" + str(current_mlp_layer-1) + "_" + str(idx) + "(v)),\n"
                
            """ the last layer should not have the l-reg, since the MLP layer apply the activation function to the layers
            in between, it is kept because the last computation apply the sigmoid to the output of the MLP layer"""
            result += "\t" + f"{mlp_bias[current_mlp_row].item():.30f}" + "\n"
            result += "\tWITH l-reg\n\tFORALL;\n"
            result += "\n"

    return result

def write_final_graph_classifier_ACR(prob_name, model):
    result = ""
    classifier_weights = extract_weights(model, "linear.weight")
    classifier_bias = extract_weights(model, "linear.bias")

    for current_row in range(len(classifier_weights)):
        result += "@final_" + prob_name + "_classifier_" + str(current_row) + "()= COMBINE\n"

        for idx in range(len(classifier_weights[current_row])):
            result += "\t(" + f"{classifier_weights[current_row][idx].item():.30f}" + "*@" + \
                prob_name + "_read_" + str(len(model.layers)) + "_" + str(idx) + "()),\n"
        
        result += "\t" + f"{classifier_bias[current_row].item():.30f}" + "\n"
        result += "\tWITH sum\n\tFORALL;\n"

        result += "\n"
    return result

def write_final_node_formula_ACR(prob_name, model, mlp, write_param=False):
    # TODO: add the mlp layer
    assert mlp == False, "MLP layer not yet supported for the final node classifier"
    result = ""
    # if hasattr(model, 'linear') and model.linear is not None:
    #     classifier_weights = extract_weights(model, "linear.weight")
    #     classifier_bias = extract_weights(model, "linear.bias")

    #     for current_row in range(len(classifier_weights)):
    #         result += prob_name + "([node]v)= COMBINE\n"

    #         for idx in range(len(classifier_weights[current_row])):
    #             result += "\t(" + f"{classifier_weights[current_row][idx].item():.30f}" + "*@" + \
    #                 prob_name + "_layer_" + str(len(model.layers)) + "_" + str(idx) + "(v)),\n"
            
    #         result += "\t" + f"{classifier_bias[current_row].item():.30f}" + "\n"
    #         result += "\tWITH l-reg\n\tFORALL;\n"
    # else:
    #     result += prob_name + "([node]v)= COMBINE\n"
    #     for l in range(len(model.layers)):
    #         result += "\t(@" + prob_name + "_layer_" + str(l) + "_"  + "(v)),\n"
    #     result += "\tWITH l-reg\n\tFORALL;\n"
    # result += "\n"


    classifier_weights = extract_weights(model, "linear.weight")
    classifier_bias = extract_weights(model, "linear.bias")

    if write_param:
        for current_row in range(len(classifier_weights)):
            result += prob_name + "([node]v)= COMBINE\n"
            for idx in range(len(classifier_weights[current_row])):
                result += "\t($final_classifier_{}_{}*@" \
                    .format(current_row, idx) + prob_name + "_layer_" + str(len(model.layers)) + "_" + str(idx) + "(v)),\n"
            result += "\t$final_classifier_bias_{}\n".format(current_row)
            result += "\tWITH l-reg\n\tFORALL;\n"
            result += "\n"
        return result

    for current_row in range(len(classifier_weights)):
        result += prob_name + "([node]v)= COMBINE\n"

        for idx in range(len(classifier_weights[current_row])):
            result += "\t(" + f"{classifier_weights[current_row][idx].item():.30f}" + "*@" + \
                prob_name + "_layer_" + str(len(model.layers)) + "_" + str(idx) + "(v)),\n"
        
        result += "\t" + f"{classifier_bias[current_row].item():.30f}" + "\n"
        result += "\tWITH l-reg\n\tFORALL;\n"

        result += "\n"
    return result

def negative_out(prob_name, class_i):
    result = "@negative_" + str(class_i) + "() = (-1 * @final_" + \
        prob_name + "_classifier_" + str(class_i) + "());\n\n"
    return result

def soft_max_out(prob_name, class_i):
    result = "soft_max_" + str(class_i) + "() = COMBINE\n"
    if class_i == 0:
        result += "\t(@final_" + prob_name + \
            "_classifier_0() + @negative_1())\n"
    elif class_i == 1:
        result += "\t(@final_" + prob_name + \
            "_classifier_1() + @negative_0())\n"
    result += "\tWITH l-reg\n\tFORALL;\n\n"
    return result