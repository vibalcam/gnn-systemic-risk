import pathlib
from typing import List, Dict, Tuple, Optional

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn.functional as F

from models.norm import GraphNorm
from models.utils import load_dict, save_dict


class FNN(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            h_features: List[int],
            out_features: int,
            activation: torch.nn.Module,
            norm_nodes: Optional[str] = True,
            dropout: float = 0.0,
            **kwargs,
    ):
        """
        Full Convolutional Neural Network

        :param in_features: input feature size
        :param h_features: list of hidden feature size
        :param out_features: out feature size        
        :param activation: If not None, applies an activation function to the updated node features
        :param norm_nodes: If not None, applies normalization to the node features
        :param dropout: dropout rate applied to intermediate layers
        """
        super(FNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()

        # input layer
        h = h_features[0]
        self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=in_features))
        self.layers.append(torch.nn.Linear(in_features, h))
        # hidden layers
        for k in h_features[1:]:
            self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=h))
            self.layers.append(torch.nn.Sequential(
                activation,
                torch.nn.Dropout(dropout),
                torch.nn.Linear(h, k),
            ))
            h = k
        # output layer
        self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=h))
        self.layers.append(torch.nn.Sequential(
            activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h, out_features),
        ))

    def forward(self, g: dgl.data.DGLDataset, feats, **kwargs):
        for n, l in zip(self.norm_layers, self.layers):
            feats = n(g, feats)
            feats = l(feats)

        return feats


class GCN(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            h_features: List[int],
            out_features: int,
            activation: torch.nn.Module,
            norm_edges: Optional[str] = 'both',
            norm_nodes: Optional[str] = None,
            dropout: float = 0.0,
            **kwargs,
    ):
        """
        Graph Convolutional Network.

        Add self loops to avoid zero-in degree

        :param in_features: input feature size
        :param h_features: list of hidden feature size
        :param out_features: out feature size        
        :param activation: If not None, applies an activation function to the updated node features
        :param norm_edges: If not None, applies normalization to the edge weights (`EdgeWeightNorm`)
        :param norm_nodes: If not None, applies normalization to the node features
        :param dropout: dropout rate applied to intermediate layers
        """
        super(GCN, self).__init__()
        self.dropout = dropout

        self.norm_layers = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        # input layer
        h = in_features
        # self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=in_features))
        # self.layers.append(dglnn.GraphConv(in_features, h, norm=norm_edges, activation=activation))
        # hidden layers
        for k in h_features:
            self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=h))
            self.layers.append(dglnn.GraphConv(h, k, norm=norm_edges, activation=activation))
            h = k
        # output layer
        self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=h))
        self.layers.append(dglnn.GraphConv(h, out_features, norm=norm_edges, activation=None))

        # todo do not finish with norm

    def forward(self, g: dgl.data.DGLDataset, feats, edge_weight=None):
        h = self.norm_layers[0](g, feats)
        h = self.layers[0](g, h, edge_weight=edge_weight)
        for n, l in zip(self.norm_layers[1:], self.layers[1:]):
            h = n(g, h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = l(g, h, edge_weight=edge_weight)
        return h


class GraphSAGE(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            h_features: List[int],
            out_features: int,
            aggregator_type: str,
            norm_edges: Optional[str] = 'both',
            norm_nodes: Optional[str] = None,
            activation: torch.nn.Module = None,
            feat_drop: float = 0.0,
            **kwargs,
    ):
        """
        GraphSAGE network

        Add self loops to avoid zero-in degree

        :param in_features: input feature size
        :param h_features: list of hidden feature size
        :param out_features: out feature size
        :param norm_edges: If not None, applies normalization to the edge weights (`EdgeWeightNorm`)
        :param norm_nodes: If not None, applies normalization to the node features
        :param activation: If not None, applies an activation function to the updated node features
        :param aggregator_type: Aggregator type to use (`mean`, `gcn`, `pool`, `lstm`).
        :param feat_drop: Dropout rate on features
        """
        super(GraphSAGE, self).__init__()

        self.norm_edges = dglnn.EdgeWeightNorm(norm_edges) if norm_edges is not None else None
        self.norm_layers = torch.nn.ModuleList()

        self.layers = torch.nn.ModuleList()
        # input layer
        h = in_features
        # self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=in_features))
        # self.layers.append(dglnn.SAGEConv(in_features, h,
        #                                   aggregator_type=aggregator_type,
        #                                   feat_drop=feat_drop,
        #                                   activation=activation))
        # hidden layers
        for k in h_features:
            self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=h))
            self.layers.append(dglnn.SAGEConv(h, k,
                                              aggregator_type=aggregator_type,
                                              feat_drop=feat_drop,
                                              activation=activation))
            h = k
        # output layer
        self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=h))
        self.layers.append(dglnn.SAGEConv(h, out_features,
                                          aggregator_type=aggregator_type,
                                          feat_drop=feat_drop,
                                          activation=None))

    def forward(self, g: dgl.data.DGLDataset, feats, edge_weight=None):
        if edge_weight is not None and self.norm_edges is not None:
            edge_weight = self.norm_edges(g, edge_weight)

        h = feats
        for n, l in zip(self.norm_layers, self.layers):
            h = n(g, h)
            h = l(g, h, edge_weight=edge_weight)
        return h


class GAT(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            h_features: List[int],
            out_features: int,
            num_heads: List[int],
            norm_nodes: Optional[str] = None,
            activation: torch.nn.Module = None,
            negative_slope: float = 0.2,
            feat_drop: float = 0.0,
            attn_drop: float = 0.0,
            residual: bool = False,
            **kwargs,
    ):
        """
        Graph Attention Network

        :param in_features: input feature size
        :param h_features: list of hidden feature size (a maximum of `len(num_heads)` elements will be used)
        :param out_features: out feature size
        :param num_heads: list of number of heads in Multi-Head Attention (a maximum of `len(h_features)` elements will be used)
        :param norm_nodes: If not None, applies normalization to the node features
        :param activation: If not None, applies an activation function to the updated node features
        :param negative_slope: LeakyReLU angle of negative slope
        :param attn_drop: Dropout rate on attention weight
        :param feat_drop: Dropout rate on features
        :param residual: If True, use residual connection
        """
        super(GAT, self).__init__()

        self.norm_layers = torch.nn.ModuleList()

        self.layers = torch.nn.ModuleList()
        # input projection (no residual)
        h = h_features[0]
        last_head = num_heads[0]
        self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=in_features))
        self.layers.append(dglnn.GATConv(
            in_feats=in_features,
            out_feats=h,
            num_heads=last_head,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=False,
            activation=activation,
        ))

        # hidden layers
        for i in range(1, min(len(h_features), len(num_heads))):
            self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=h * last_head))
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(dglnn.GATConv(
                in_feats=h * last_head,
                out_feats=h_features[i],
                num_heads=num_heads[i],
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=activation,
            ))
            h = h_features[i]
            last_head = num_heads[i]

        # output projection
        self.norm_layers.append(GraphNorm(norm_nodes, hidden_dim=h * last_head))
        self.layers.append(dglnn.GATConv(
            in_feats=h * last_head,
            out_feats=out_features,
            num_heads=1,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            activation=None,
        ))

    def forward(self, g: dgl.data.DGLDataset, feats, **kwargs):
        h = feats
        for n, l in zip(self.norm_layers, self.layers):
            h = n(g, h)
            # flatten all except dimension 0 (number of nodes)
            h = l(g, h).flatten(1)
        return h


MODEL_CLASS = {
    'gcn': GCN,
    'sage': GraphSAGE,
    'gat': GAT,
    'fnn': FNN,
}
MODEL_CLASS_KEY = 'model_class'
FOLDER_PATH_KEY = 'path_name'


def save_model(model: torch.nn.Module, folder: str, model_name: str, param_dicts: Dict = None,
               save_model: bool = True) -> None:
    """
    Saves the model so it can be loaded after

    :param model_name: name of the model to be saved (non including extension)
    :param folder: path of the folder where to save the model
    :param param_dicts: dictionary of the model parameters that can later be used to load it
    :param model: model to be saved
    :param save_model: If true the model and dictionary will be saved, otherwise only the dictionary will be saved
    """
    # create folder if it does not exist
    folder_path = f"{folder}/{model_name}"
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    # save model
    if save_model:
        torch.save(model.state_dict(), f"{folder_path}/{model_name}.th")

    # save dict
    if param_dicts is None:
        param_dicts = {}
    
    # get class of the model
    model_class = None
    for k, v in MODEL_CLASS.items():
        if isinstance(model, v):
            model_class = k
            break
    if model_class is None:
        raise Exception("Model class unknown")
    param_dicts[MODEL_CLASS_KEY] = model_class

    # save the dictionary as plain text and pickle
    save_dict(param_dicts, f"{folder_path}/{model_name}.dict", as_str=True)
    save_dict(param_dicts, f"{folder_path}/{model_name}.dict.pickle", as_str=False)


def load_model(folder_path: pathlib.Path, model_class: Optional[str] = None) -> Tuple[torch.nn.Module, Dict]:
    """
    Loads a model that has been previously saved using its name (model th and dict must have that same name)

    :param folder_path: folder path of the model to be loaded
    :param model_class: one of the model classes in `MODEL_CLASS` dict. If none, it is obtained from the dictionary
    :return: the loaded model and the dictionary of parameters
    """
    # todo so it does not need to have the same name
    path = f"{folder_path.absolute()}/{folder_path.name}"
    # use pickle dictionary
    dict_model = load_dict(f"{path}.dict.pickle")

    # get model class
    if model_class is None:
        model_class = dict_model.get(MODEL_CLASS_KEY)

    # set folder path
    dict_model[FOLDER_PATH_KEY] = folder_path.name

    return load_model_data(MODEL_CLASS[model_class](**dict_model), f"{path}.th"), dict_model


def load_model_data(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """
    Loads a model than has been previously saved

    :param model_path: path from where to load model
    :param model: model into which to load the saved model
    :return: the loaded model
    """
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model
