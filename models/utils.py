import pickle
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
from sklearn.metrics import matthews_corrcoef, mean_squared_error

QUANTILES = [0.75, 0.5, 0.25]
NODE_ATTR = [
    'assets',
    'liabilities',
    'buffer',
    'weights',
]
NODES_FILENAME = 'nodes.csv'
NETWORK_FILENAME = 'network.csv'


class ContagionDataset(dgl.data.DGLDataset):
    """Class that represents a dataset to train the classifier"""

    def __init__(
            self,
            raw_dir: str = './data',
            drop_edges: float = 0,
            add_self_loop: bool = True,
            sets_lengths: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            seed: int = 123,
            target: str = 'additional_stress',
            node_attributes: List[str] = NODE_ATTR,
    ):
        """
        Initializer for the dataset
        :param raw_dir: directory where the input data is stored
        :param drop_edges: percentage of edges to remove. Value in [0,1]
        :param add_self_loop: If true, it adds non duplicated self loops
        :param sets_lengths: tuple with percentage of train, validation and test samples
        :param seed: seed to randomly generate train, valid and test sets
        :param target: column to use as target for quantile calculation
        :param node_features: list of names of the columns to use as node features
        """
        if not (0 <= drop_edges <= 1):
            raise Exception("drop_edges must be a value in [0,1]")

        self.target_col = target
        self.sets_lengths = sets_lengths
        self.drop_edges = drop_edges
        self.add_self_loop = add_self_loop
        self.random_generator = torch.manual_seed(seed)
        # todo reset random seed

        # todo change this
        self.num_classes = len(QUANTILES) + 1
        self.node_attributes = node_attributes
        self.num_node_features = len(self.node_attributes)

        super().__init__(raw_dir=raw_dir, name='contagion', verbose=False)

    def process(self):
        # todo change location of files, for loop folder raw_dir
        # pathlib.Path(self.raw_dir)
        self.graphs = []
        self.targets = []
        self.node_features = []

        # LOAD DATA

        nodes = pd.read_csv(f'{self.raw_dir}/{NODES_FILENAME}', index_col=0).set_index('bank')
        network = pd.read_csv(f'{self.raw_dir}/{NETWORK_FILENAME}', index_col=0)
        # create networkx graph from adjacency matrix
        graph = nx.convert_matrix.from_pandas_adjacency(network, create_using=nx.DiGraph)

        # GET TARGET

        quant = nodes[self.target_col].quantile(QUANTILES)
        is_quant = pd.DataFrame()
        free = np.ones(nodes.shape[0]).astype(bool)
        # get those higher than percentile and make them unavailable
        for k, v in quant.iteritems():
            is_quant[k] = np.logical_and(nodes[self.target_col] >= v, free)
            free = np.logical_and(free, np.logical_not(is_quant[k]))
        # last quantile are those still available
        is_quant[0.0] = free
        # from one_hot to labels
        is_quant_np = is_quant.to_numpy().astype(float)
        target_np = is_quant_np.argmax(1)
        self.targets.append(target_np)
        # to dataframe for to_dict in networkx
        is_quant = pd.DataFrame(data=target_np, index=is_quant.index, columns=['label'])

        # ADD NODE DATA TO GRAPH

        # add features
        nodes_features = nodes[self.node_attributes]
        self.node_features.append(nodes_features)
        # add features to node in form: {node:{"feat":values}}
        nx.set_node_attributes(graph, {k: {"feat": torch.as_tensor(v, dtype=torch.float)} for k, v in
                                       nodes_features.T.to_dict('list').items()})
        # add target
        nx.set_node_attributes(graph, is_quant.to_dict('index'))

        # CREATE DGL GRAPH
        graph_dgl = dgl.from_networkx(graph, node_attrs=['feat', 'label'], edge_attrs=['weight'])

        # ADD TRAIN,VALIDATION,TEST MASKS

        n_nodes = graph_dgl.num_nodes()
        n_train, n_val = (int(n_nodes * k) for k in self.sets_lengths[:2])
        train_mask, val_mask, test_mask = [torch.zeros(n_nodes, dtype=torch.bool) for k in range(3)]
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        # shuffle and set mask in nodes
        idx = torch.randperm(n_nodes, generator=self.random_generator)
        graph_dgl.ndata['train_mask'] = train_mask[idx]
        graph_dgl.ndata['val_mask'] = val_mask[idx]
        graph_dgl.ndata['test_mask'] = test_mask[idx]

        # add to list
        self.graphs.append(graph_dgl)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i) -> dgl.DGLGraph:
        k = self.graphs[i]
        # implementing DropEdges by randomly removing edges from graph
        if self.drop_edges > 0:
            k = copy.deepcopy(k)
            n_remove = int(k.num_edges() * self.drop_edges)
            k.remove_edges(torch.randint(k.num_edges(), size=(n_remove,), generator=self.random_generator))

        # add self loops
        if self.add_self_loop:
            k = k.remove_self_loop().add_self_loop()

        # todo transform labels

        return k


def labels_to_percentile(labels:torch.Tensor, n_classes:int, random_u:bool = True) -> torch.Tensor:
    """
    Transforms a tensor labels with the class importance to a [0,1] value corresponding to its approximate percentile

    :param labels: labels with the class importance
    :param n_classes: number of classes
    :param random_u: if true, it will apply a random uniform to the approximate percentile
    """

    pass


def percentile_to_labels(x:torch.Tensor, n_classes:int) -> torch.Tensor:
    """
todo finish

    :param x: 
    :param n_classes: number of classes
    """
    pass


class ConfusionMatrix:
    """
    Class that represents a confusion matrix. 
    
    Cij is equal to the number of observations known to be in class i and predicted in class j
    """

    def _make(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Returns the confusion matrix of the given predicted and labels values
        :param preds: predicted values (B)
        :param labels: true values (B)
        :return: (size,size) confusion matrix of `size` classes
        """
        matrix = torch.zeros(self.size, self.size, dtype=torch.float)
        for t, p in zip(labels.reshape(-1).long(), preds.reshape(-1).long()):
            matrix[t, p] += 1
        return matrix

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size, dtype=torch.float)
        self.preds = None
        self.labels = None
        self.size = size

    def __repr__(self) -> str:
        return self.matrix.__repr__

    def add(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        :param preds: predicted values (B)
        :param labels: true values (B)
        """
        preds = preds.reshape(-1).cpu().detach().clone()
        labels = labels.reshape(-1).cpu().detach().clone()
        self.matrix += self._make(preds, labels)
        self.preds = torch.cat((self.preds, preds), dim=0) if self.preds is not None else preds
        self.labels = torch.cat((self.labels, labels), dim=0) if self.labels is not None else labels

    @property
    def matthews_corrcoef(self):
        """Matthews correlation coefficient (MCC)"""
        return matthews_corrcoef(y_true=self.labels.numpy(),y_pred=self.preds.numpy())

    @property
    def rmse(self):
        return mean_squared_error(y_true=self.labels,y_pred=self.preds, squared=False)

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return (true_pos.sum() / (self.matrix.sum() + 1e-5)).item()

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean().item()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)

    @property
    def normalize(self):
        return self.matrix / (self.matrix.sum() + 1e-5)

    def visualize(self, normalize: bool = False):
        """
        Visualize confusion matrix
        :param normalize: whether to normalize the matrix by the total amount of samples
        """
        plt.figure(figsize=(15, 10))

        matrix = self.normalize.numpy() if normalize else self.matrix.numpy()

        df_cm = pd.DataFrame(matrix).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return plt

    def __repr__(self):
        return self.visualize()


def save_dict(d: Dict, path: str, as_str: bool = False) -> None:
    """
    Saves a dictionary to a file in plain text
    :param d: dictionary to save
    :param path: path of the file where the dictionary will be saved
    :param as_str: If true, it will save as a string. If false, it will use pickle
    """
    if as_str:
        with open(path, 'w', encoding="utf-8") as file:
            file.write(str(d))
    else:
        with open(path, 'wb') as file:
            pickle.dump(d, file)


def load_dict(path: str) -> Dict:
    """
    Loads a dictionary from a file (plain text or pickle)
    :param path: path where the dictionary was saved

    :return: the loaded dictionary
    """
    with open(path, 'rb') as file:
        try:
            return pickle.load(file)
        except pickle.UnpicklingError as e:
            # print(e)
            pass

    with open(path, 'r', encoding="utf-8") as file:
        from ast import literal_eval
        s = file.read()
        return dict(literal_eval(s))
        # try:
        #     return dict(literal_eval(s))
        # except SyntaxError:
        #     pass


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
