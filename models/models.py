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


def load_model(folder_path: pathlib.Path) -> Tuple[torch.nn.Module, Dict]:
    """
    Loads a model that has been previously saved using its name (model th and dict must have that same name)
    Only works for StateActionModel
    :param folder_path: folder path of the model to be loaded
    :return: the loaded model and the dictionary of parameters
    """
    path = f"{folder_path.absolute()}/{folder_path.name}"
    dict_model = load_dict(f"{path}.dict")
    return load_model_data(StateActionModel(**dict_model), f"{path}.th"), dict_model


def load_model_data(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """
    Loads a model than has been previously saved
    :param model_path: path from where to load model
    :param model: model into which to load the saved model
    :return: the loaded model
    """
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model
