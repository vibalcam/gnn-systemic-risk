from typing import List, Dict, Tuple

import torch
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F

from models.utils import load_dict, save_dict
import pathlib

# todo add norm options

class GCN(torch.nn.Module):
    def __init__(
        self, 
        in_features:int, 
        h_features:List[int], 
        out_features:int,
        activation:torch.nn.Module,
        dropout:float=0.0
    ):
        """
        
        """
        super(GCN, self).__init__()
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        # input layer
        h = h_features[0]
        self.layers.append(dglnn.GraphConv(in_features, h, activation=activation))
        # hidden layers
        for k in h_features[1:]:
            self.layers.append(dglnn.GraphConv(h, k, activation=activation))
            h = k
        # output layer
        self.layers.append(dglnn.GraphConv(h, out_features, activation=None))
    
    def forward(self, g:dgl.data.DGLDataset, feats):
        # todo add self loops to avoid zero-in degree, move to dataset 
        # (https://docs.dgl.ai/en/0.6.x/api/python/nn.pytorch.html#graphconv)
        g = g.remove_self_loop().add_self_loop()

        h = self.layers[0](g, feats)
        for l in self.layers[1:]:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = l(g,h)
        return h
    
# # Create the model with given dimensions
# model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)


class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_features:int,
        h_features:List[int], 
        out_features:int,
        activation:torch.nn.Module,
        aggregator_type,
        feat_drop:float=0.0,
    ):
        """
        
        """
        super(GraphSAGE, self).__init__()
        self.layers = torch.nn.ModuleList()
        # self.norm_layers = torch.nn.ModuleList()

        # input layer
        # normalization in SAGEConv module is applied after
        h = h_features[0]
        self.layers.append(dglnn.SAGEConv(in_features, h, aggregator_type,norm=None, feat_drop=feat_drop, activation=activation))
        # hidden layers
        for k in h_features[1:]:
            self.layers.append(dglnn.SAGEConv(h, k, aggregator_type, feat_drop=feat_drop, activation=activation))
            h = k
        # output layer
        self.layers.append(dglnn.SAGEConv(h, out_features, aggregator_type, feat_drop=feat_drop, activation=None))

    def forward(self, g:dgl.data.DGLDataset, feats):
        h = feats
        for layer in self.layers:
            h = layer(self.g, h)
        return h

# GAT -> 

# try batch normalization


def save_model(model: torch.nn.Module, folder: str, model_name: str, param_dicts: Dict = None) -> None:
    """
    Saves the model so it can be loaded after
    :param model_name: name of the model to be saved (non including extension)
    :param folder: path of the folder where to save the model
    :param param_dicts: dictionary of the model parameters that can later be used to load it
    :param model: model to be saved
    """
    # create folder if it does not exist
    folder_path = f"{folder}/{model_name}"
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    # save model
    torch.save(model.state_dict(), f"{folder_path}/{model_name}.th")
    # save dict
    if param_dicts is not None:
        save_dict(param_dicts, f"{folder_path}/{model_name}.dict")


def load_model(model_class, folder_path: pathlib.Path) -> Tuple[torch.nn.Module, Dict]:
    """
    Loads a model that has been previously saved using its name (model th and dict must have that same name)
    Only works for StateActionModel
    :param folder_path: folder path of the model to be loaded
    :return: the loaded model and the dictionary of parameters
    """
    path = f"{folder_path.absolute()}/{folder_path.name}"
    dict_model = load_dict(f"{path}.dict")
    return load_model_data(model_class(**dict_model), f"{path}.th"), dict_model


def load_model_data(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """
    Loads a model than has been previously saved
    :param model_path: path from where to load model
    :param model: model into which to load the saved model
    :return: the loaded model
    """
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model
