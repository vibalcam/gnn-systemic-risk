from typing import List, Any, Dict, Tuple
import copy

import pandas as pd
import dgl
import torch
import numpy as np
import networkx as nx
import pathlib
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

TARGET_COLUMN = 'additional_stress'
QUANTILES = [0.75, 0.5, 0.25]
NODE_ATTR = [
    'assets',
    'liabilities',
    # 'buffer',
]


class ContagionDataset(dgl.data.DGLDataset):
    """Class that represents a dataset to train the classifier"""

    def __init__(self,raw_dir:str='./data', drop_edges:float = 0,
                sets_lengths:Tuple[float,float,float]=(0.8, 0.1, 0.1), seed:int=123):
        """
        Initializer for the dataset
        :param raw_dir: directory where the input data is stored
        :param drop_edges: percentage of edges to remove. Value in [0,1]
        :param sets_lengths: tuple with percentage of train, validation and test samples
        :param seed: seed to randomly generate train, valid and test sets
        """
        if not (0 <= drop_edges <= 1):
            raise Exception("drop_edges must be a value in [0,1]")
        
        super().__init__(raw_dir=raw_dir, name='contagion', verbose=True)
        self.sets_lengths = sets_lengths
        self.drop_edges = drop_edges
        self.random_generator = torch.manual_seed(seed)
        

        # todo change this
        self.num_classes = len(QUANTILES)
        self.node_features = len(NODE_ATTR)

    def process(self):
        # todo change location of files, for loop folder raw_dir
        # pathlib.Path(self.raw_dir)
        self.graphs = []
        
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
        idx = torch.randperm(n_nodes, generator=self.random_generator)
        train_mask, val_mask[idx], test_mask[idx] = train_mask[idx], val_mask[idx], test_mask[idx]
        # set mask in nodes
        graph_dgl.ndata['train_mask'] = train_mask
        graph_dgl.ndata['val_mask'] = val_mask
        graph_dgl.ndata['test_mask'] = test_mask

        # add to list
        self.graphs.append(graph_dgl)


    def __len__(self):
        return len(self.graphs)

    def __getitem__(self,i) -> dgl.DGLGraph:
        k = self.graphs[i]
        # implementing DropEdges by randomly removing edges from graph
        if self.drop_edges > 0:
            k = copy.deepcopy(k)
            n_remove = int(k.num_edges() * self.drop_edges)
            k.remove_edges(torch.randint(k.num_edges(), size=(n_remove,), generator=self.random_generator))

        return k


class ConfusionMatrix:
    """
    Class that represents a confusion matrix. 
    
    Cij is equal to the number of observations known to be in class i and predicted in class j
    """
    def _make(self, preds:torch.Tensor, labels:torch.Tensor) -> torch.Tensor:
        """
        Returns the confusion matrix of the given predicted and labels values
        :param preds: predicted values (B)
        :param labels: true values (B)
        :return: (size,size) confusion matrix of `size` classes
        """
        matrix = torch.zeros(self.size, self.size, dtype=torch.float)
        for t, p in zip(labels.reshape(-1).long().cpu().detach(), preds.reshape(-1).long().cpu().detach()):
            matrix[t, p] += 1
        return matrix

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size, dtype=torch.float)
        self.size = size

    def __repr__(self) -> str:
        return self.matrix.__repr__

    def add(self, preds:torch.Tensor, labels:torch.Tensor) -> None:
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        :param preds: predicted values (B)
        :param labels: true values (B)
        """
        self.matrix += self._make(preds, labels)

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)

    @property
    def normalize(self):
        return self.matrix / (self.matrix.sum() + 1e-5)

    def visualize(self, normalize:bool=False):
        """
        Visualize confusion matrix
        :param normalize: whether to normalize the matrix by the total amount of samples
        """
        plt.figure(figsize=(15,10))

        matrix = self.normalize.numpy() if normalize else self.matrix.numpy()

        df_cm = pd.DataFrame(matrix).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        return plt


def save_dict(d: Dict, path: str) -> None:
    """
    Saves a dictionary to a file in plain text
    :param d: dictionary to save
    :param path: path of the file where the dictionary will be saved
    """
    with open(path, 'w', encoding="utf-8") as file:
        file.write(str(d))


def load_dict(path: str) -> Dict:
    """
    Loads a dictionary from a file in plain text
    :param path: path where the dictionary was saved
    :return: the loaded dictionary
    """
    with open(path, 'r', encoding="utf-8") as file:
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
    """
    Loads a list from a file in plain text
    :param path: path where the list was saved
    :return: the loaded list
    """
    with open(path, 'r', encoding="utf-8") as file:
        from ast import literal_eval
        loaded = list(literal_eval(file.read()))
    return loaded


if __name__ == '__main__':
    pass