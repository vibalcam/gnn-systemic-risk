from typing import List, Any, Dict, Tuple

import pandas as pd
import dgl
import torch
import numpy as np
import networkx as nx
import pathlib
from glob import glob

TARGET_COLUMN = 'additional_stress'
QUANTILES = [0.75, 0.5, 0.25]
NODE_ATTR = [
    'assets',
    'liabilities',
    # 'buffer',
]


class ContagionDataset(dgl.data.DGLDataset):
    """Class that represents a dataset to train the classifier"""

    def __init__(self, sets_lengths:Tuple[float,float,float]=(0.8, 0.1, 0.1), seed:int=123):
        """
        Initializer for the dataset
        :param sets_lengths: tuple with percentage of train, validation and test samples
        :param seed: seed to randomly generate train, valid and test sets
        """
        
        super().__init__(name='contagion', verbose=True)
        self.sets_lengths = sets_lengths
        torch.manual_seed(seed)

    def process(self):
        # todo change location of files, for loop folder raw_dir
        # pathlib.Path(self.raw_dir)
        
        # LOAD DATA
        nodes = pd.read_csv('nodes.csv', index_col=0).set_index('bank')
        network = pd.read_csv('network.csv', index_col=0)
        # create networkx graph from adjacency matrix
        graph = nx.convert_matrix.from_pandas_adjacency(network, create_using=nx.DiGraph)

        # GET TARGET
        quant = nodes[TARGET_COLUMN].quantile(QUANTILES)
        is_quant = pd.DataFrame()
        free = np.ones(nodes.shape[0]).astype(bool)
        # get those higher than percentile and make them unavailable
        for k,v in quant.iteritems():
            is_quant[k] = np.logical_and(nodes[TARGET_COLUMN] >= v, free)
            free = np.logical_and(free, np.logical_not(is_quant[k]))
        # last quantile are those still available
        is_quant[0.0] = free
        # from one_hot to labels
        is_quant_np = is_quant.to_numpy().astype(float)
        target_np = is_quant_np.argmax(1)
        # to dataframe for to_dict in networkx
        is_quant = pd.DataFrame(data=target_np, index=is_quant.index, columns=['label'])

        # ADD NODE DATA TO GRAPH
        # add features
        nodes_features = nodes[NODE_ATTR]
        # add features to node in form: {node:{"feat":values}}
        nx.set_node_attributes(graph, {k:{"feat":torch.as_tensor(v, dtype=torch.float)} for k,v in nodes_features.T.to_dict('list').items()})
        # add target
        nx.set_node_attributes(graph, is_quant.to_dict('index'))

        # CREATE DGL GRAPH
        graph_dgl = dgl.from_networkx(graph,node_attrs=['feat', 'label'],edge_attrs=['weight'])

        # ADD TRAIN,VALIDATION,TEST MASKS
        n_nodes = graph_dgl.num_nodes()
        n_train, n_val = (int(n_nodes * k) for k in self.sets_lengths[:2])
        train_mask, val_mask, test_mask = [torch.zeros(n_nodes, dtype=torch.bool) for k in range(3)]
        train_mask[:n_train] = True
        val_mask[n_train:n_train+n_val] = True
        test_mask[n_train+n_val:] = True
        # shuffle
        idx = torch.randperm(n_nodes)
        train_mask, val_mask[idx], test_mask[idx] = train_mask[idx], val_mask[idx], test_mask[idx]
        # set mask in nodes
        graph_dgl.ndata['train_mask'] = train_mask
        graph_dgl.ndata['val_mask'] = val_mask
        graph_dgl.ndata['test_mask'] = test_mask


    def __len__(self):
        return len(self.graphs)

    def __getitem__(self,i) -> dgl.DGLGraph:
        return self.graphs[i]


def accuracy(predicted: torch.Tensor, label: torch.Tensor, mean: bool = True):
    """
    Calculates the accuracy of the prediction and returns a numpy number.
    It considers predicted to be class 1 if probability is higher than 0.5

    :param mean: true to return the mean, false to return an array
    :param predicted: torch.Tensor: the input prediction
    :param label: torch.Tensor: the real label
    :param mean: bool:  (Default value = True) whether to return the mean or not reduced
    :returns: returns the accuracy of the prediction (between 0 and 1), in the cpu and detached as numpy
    """

    correct = (predicted == label).float()
    if mean:
        return correct.mean().cpu().detach().numpy()
    else:
        return correct.cpu().detach().numpy()

# todo añadir encoding

def save_dict(d: Dict, path: str) -> None:
    """Saves a dictionary to a file in plain text

    :param d: Dict: dictionary to save
    :param path: str: path of the file where the dictionary will be saved
    """

    with open(path, 'w') as file:
        file.write(str(d))


def load_dict(path: str) -> Dict:
    """Loads a dictionary from a file in plain text

    :param path: str: path where the dictionary was saved
    :returns: the loaded dictionary
    """

    with open(path, 'r') as file:
        from ast import literal_eval
        loaded = dict(literal_eval(file.read()))
    return loaded


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_list(path: str) -> List:
    """Loads a list from a file in plain text

    :param path: str: path where the list was saved
    :returns: the loaded list
    """

    with open(path, 'r') as file:
        from ast import literal_eval
        loaded = list(literal_eval(file.read()))
    return loaded
