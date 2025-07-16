
import torch

from python.gnn import *

######
# PYTHON CONFIG PATH FOR PRIMULA
######

def load_model():
    model = MYACRGnnNodePrim(
        input_dim=5,
        hidden_dim=[5],
        num_layers=1,        
        final_read="add",
        num_classes=1
    ).to("cpu")
    model.load_state_dict(torch.load(
        '/PATH/TO/WEIGHTS.pt',
        map_location="cpu",
        weights_only=True
    ))
    model.eval()
    return model
