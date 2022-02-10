from typing import List, Dict

import torch
import dgl

from models.utils import load_dict




class ContagionNN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()


        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h






def save_model(model: torch.nn.Module, filename: str = None) -> None:
    """
    Saves the model so it can be loaded after
    :param filename: filename where the model should be saved (non including extension)
    :param model: model to be saved
    """
    from torch import save
    from os import path

    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), f"{filename if filename is not None else n}.th")
    raise Exception(f"Model type {type(model)} not supported")


def load_model_from_name(model_name):
    """
    Loads a model that has been previously saved using its name (model th and dict must have that same name)
    :param model_name: name of the model to load
    :return: the loaded model
    """
    dict_model = load_dict(f"{model_name}.dict")
    return load_model(f"{model_name}.th", StateActionModel(**dict_model))


def load_model(model_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """
    Loads a model than has been previously saved
    :param model_path: path from where to load model
    :param model: model into which to load the saved model
    :return: the loaded model
    """
    from torch import load
    model.load_state_dict(load(model_path, map_location='cpu'))
    return model
