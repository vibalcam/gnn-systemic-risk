import copy
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Union

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import matthews_corrcoef, mean_squared_error, mean_absolute_error

# N_TILES = [0.75, 0.5, 0.25]
NODE_ATTR = [
    'assets',
    'liabilities',
    'buffer',
    'weights',
]
NODES_FILENAME = 'nodes.csv'
NETWORK_FILENAME = 'network.csv'


class ContagionDataset(dgl.data.DGLDataset):
    """
    Class that represents a dataset to train the classifier
    The graphs have the following node attributes: `feat`, `label`, `perc`, `id`, `train_mask`, `val_mask` and `test_mask`.
    The graphs have the following edge attributes: `weight`.
    """

    def __init__(
            self,
            raw_dir: str = './data',
            drop_edges: float = 0,
            add_self_loop: bool = True,
            sets_lengths: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            target: str = 'additional_stress',
            node_attributes: List[str] = NODE_ATTR,
            num_classes: int = 4,
            seed:int = 4444
    ):
        """
        Initializer for the dataset. 

        The parameters `sets_lengths, target, node_attributes`

        :param raw_dir: directory where the input data is stored
        :param drop_edges: percentage of edges to remove. Value in [0,1]
        :param add_self_loop: If true, it adds non duplicated self loops
        :param sets_lengths: tuple with percentage of train, validation and test samples
        :param target: column to use as target for quantile calculation
        :param node_attributes: list of names of the columns to use as node features
        :param num_classes: number of classes, n-tiles which the labels will represent
        """
        if not (0 <= drop_edges <= 1):
            raise Exception("drop_edges must be a value in [0,1]")

        self.target_col = target
        self.sets_lengths = sets_lengths
        self.drop_edges = drop_edges
        self.add_self_loop = add_self_loop
        self.seed = seed

        # initialized in process
        self.random_generator = None

        # self.num_classes = len(N_TILES) + 1
        self.num_classes = num_classes
        self.node_attributes = node_attributes

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
        # 0 is the highest class (last n-tile), n_classes - 1 the lowest (1st n-tile)

        # pd.qcut returns the corresponding n-tile
        is_quant = (self.num_classes - 1) - pd.qcut(nodes[self.target_col], self.num_classes, labels=False)
        target_np = is_quant.to_numpy().astype(int)

        # quant = nodes[self.target_col].quantile(N_TILES)
        # is_quant = pd.DataFrame()
        # free = np.ones(nodes.shape[0]).astype(bool)
        # # get those higher than percentile and make them unavailable
        # for k, v in quant.iteritems():
        #     is_quant[k] = np.logical_and(nodes[self.target_col] >= v, free)
        #     free = np.logical_and(free, np.logical_not(is_quant[k]))
        # # last quantile are those still available
        # is_quant[0.0] = free
        # # from one_hot to labels
        # is_quant_np = is_quant.to_numpy().astype(float)
        # target_np = is_quant_np.argmax(1)

        self.targets.append(target_np)

        # ADD NODE DATA TO GRAPH

        # add features
        nodes_features = nodes[self.node_attributes]
        self.node_features.append(nodes_features)
        # add features to node in form: {node:{"feat":values}}
        nx.set_node_attributes(graph, {k: {"feat": torch.as_tensor(v, dtype=torch.float)} for k, v in
                                       nodes_features.T.to_dict('list').items()})
        # add target
        is_quant = pd.DataFrame(data=target_np, index=is_quant.index, columns=['label'])
        nx.set_node_attributes(graph, is_quant.to_dict('index'))

        # add percentiles [0,1]
        percentiles = pd.qcut(nodes[self.target_col], 100, labels=False) / 100
        percentiles = pd.DataFrame(data=percentiles.to_numpy().astype(float), index=percentiles.index, columns=['perc'])
        nx.set_node_attributes(graph, percentiles.to_dict('index'))

        # add bank ids
        nx.set_node_attributes(graph, {k: {'id': int(k[1:])} for k in is_quant.index})

        self.nx_graph = graph

        # CREATE DGL GRAPH
        graph_dgl = dgl.from_networkx(graph, node_attrs=['feat', 'label', 'perc', 'id'], edge_attrs=['weight'])

        # ADD TRAIN,VALIDATION,TEST MASKS

        n_nodes = graph_dgl.num_nodes()
        n_train, n_val = (int(n_nodes * k) for k in self.sets_lengths[:2])
        train_mask, val_mask, test_mask = [torch.zeros(n_nodes, dtype=torch.bool) for k in range(3)]
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        # shuffle and set mask in nodes
        # random generator for dataset
        self.random_generator = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(n_nodes, generator=self.random_generator)
        graph_dgl.ndata['train_mask'] = train_mask[idx]
        graph_dgl.ndata['val_mask'] = val_mask[idx]
        graph_dgl.ndata['test_mask'] = test_mask[idx]

        # add to list
        self.graphs.append(graph_dgl)

    @property
    def num_node_features(self):
        return len(self.node_attributes)

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

        return k

    def plot_classes(self, name, network='eur'):
        # plt.switch_backend('pgf')
        graph = self.nx_graph

        legend_colors = ['#FF0000', '#FFA500', '#7FFF00', '#008000']
        legend_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
        node_color = [legend_colors[graph.nodes[node]['label']] for node in graph.nodes]

        # Set node size based on degree
        node_size = [10*d for n, d in graph.degree()]

        # Use force-directed layout algorithm to position nodes
        if network == 'eur':
            pos = nx.spring_layout(graph, iterations=200, k=120, scale=2, seed=self.seed)
        else:
            pos = nx.spring_layout(graph, iterations=200, k=0.5, scale=1, seed=self.seed)

        # Plot the graph
        fig = plt.figure(figsize=(15, 15))
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=0.1, alpha=0.5)
        plt.axis('off')
        legend_handles = []
        for color, label in zip(legend_colors, legend_labels):
            legend_handles.append(mpatches.Patch(color=color, label=label))

        plt.legend(handles=legend_handles, title='Node colors', loc='best')
        fig.savefig(name, format='svg')
        # fig.savefig('my_graph.pgf')
        # plt.show()

        return fig

    def plot_train_test(self, name, network='eur'):
        graph = self.nx_graph

        legend_colors = ['#B9B9B9', 'red']
        legend_labels = ['Test', 'Train&Validation']
        mask = torch.logical_or(self.graphs[0].ndata['train_mask'], self.graphs[0].ndata['val_mask']).int()
        node_color = [legend_colors[self.graphs[0].ndata['train_mask'][idx]] for idx, node in enumerate(graph.nodes)]

        # Set node size based on degree
        node_size = [10*d for n, d in graph.degree()]

        # Use force-directed layout algorithm to position nodes
        if network == 'eur':
            pos = nx.spring_layout(graph, iterations=200, k=120, scale=2, seed=self.seed)
        else:
            pos = nx.spring_layout(graph, iterations=200, k=0.5, scale=1, seed=self.seed)

        # Plot the graph
        fig = plt.figure(figsize=(15, 15))
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=0.1, alpha=0.5)
        plt.axis('off')
        legend_handles = []
        for color, label in zip(legend_colors, legend_labels):
            legend_handles.append(mpatches.Patch(color=color, label=label))

        plt.legend(handles=legend_handles, title='Node colors', loc='best')
        fig.savefig(name, format='svg')

        return fig


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

    def __init__(self, size=5, name: str = ''):
        """
        This class builds and updates a confusion matrix.

        :param size: the number of classes to consider
        :param name: name of the confusion matrix
        """
        self.matrix = torch.zeros(size, size, dtype=torch.float)
        self.preds = None
        self.labels = None
        self.pred_perc = None
        self.true_percentiles = None
        self.name = name

    def bootstrap(self, metric, n=1000, seed=12345, reg=False):
        """
        Bootstraps the confusion matrix to get a confidence interval for the given metric

        :param n: number of bootstrap samples
        :param metric: metric to compute
        :param seed: random seed
        :param significance: significance level
        :return: (lower, upper, mean)
        """
        if reg:
            p = self.pred_perc
            l = self.true_percentiles
        else:
            p = self.preds
            l = self.labels

        torch.manual_seed(seed)
        accs = []
        for _ in range(n):
            idx = np.random.choice(p.shape[0], size=p.shape[0], replace=True).astype(int)
            preds = p[idx]
            labels = l[idx]
            accs.append(metric(y_true=labels.numpy(), y_pred=preds.numpy()))
        accs = np.asarray(accs)

        return accs

    def __repr__(self) -> str:
        return self.matrix.numpy().__repr__()

    def add(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs) -> None:
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
    def size(self):
        return self.matrix.shape[0]

    @property
    def matthews_corrcoef(self):
        """Matthews correlation coefficient (MCC)"""
        return matthews_corrcoef(y_true=self.labels.numpy(), y_pred=self.preds.numpy())

    @property
    def mae(self):
        return mean_absolute_error(y_true=self.labels.numpy(), y_pred=self.preds.numpy())

    @property
    def rmse(self):
        return mean_squared_error(y_true=self.labels.numpy(), y_pred=self.preds.numpy(), squared=False)

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
        return self.matrix / (self.matrix.sum() + 1e-6)

    def visualize(self, normalize: bool = False):
        """
        Visualize confusion matrix
        :param normalize: whether to normalize the matrix by the total amount of samples
        """
        plt.figure(figsize=(15, 10))

        matrix = self.normalize.numpy() * 100 if normalize else self.matrix.numpy()

        df_cm = pd.DataFrame(matrix).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return plt


class ClassConfusionMatrix(ConfusionMatrix):
    def __init__(self, size=5, name: str = ''):
        super().__init__(size, name)
        # self.true_percentiles = None
        # self.pred_perc = None

    def add(self, preds: torch.Tensor, labels: torch.Tensor, true_percentiles: torch.Tensor) -> None:
        super().add(preds, labels)

        true_percentiles = true_percentiles.reshape(-1).cpu().detach().clone()
        self.true_percentiles = torch.cat((self.true_percentiles, true_percentiles),
                                          dim=0) if self.true_percentiles is not None else true_percentiles

        # use mean/median (the middle value since uniform distribution) as predicted percentile, since it minimizes the MSE/MAE
        preds = preds.reshape(-1).cpu().detach().clone()
        pred_perc = 1 - (preds / self.size + 1 / (2 * self.size))
        self.pred_perc = torch.cat((self.pred_perc, pred_perc),
                                   dim=0) if self.pred_perc is not None else pred_perc

    @property
    def rmse_percentiles(self):
        return mean_squared_error(y_true=self.true_percentiles, y_pred=self.pred_perc, squared=False)

    @property
    def mae_percentiles(self):
        return mean_absolute_error(y_true=self.true_percentiles, y_pred=self.pred_perc)


class PercentilesConfusionMatrix(ConfusionMatrix):
    def __init__(self, size=5, name: str = ''):
        """
        This class builds and updates a confusion matrix.

        :param size: the number of classes to consider
        :param name: name of the confusion matrix
        :param base_n: if true, it will expect to receive predictions in range `[0,self.size]`,
                        otherwise, it will expect them in range `[0,self.size-1]`
        """
        super().__init__(size, name)
        self.perc = None
        # self.true_percentiles = None

    def bootstrap(self, *args, **kwargs):
        self.pred_perc = self.perc
        return super().bootstrap(*args, **kwargs)

    def add(self, preds: torch.Tensor, labels: torch.Tensor, true_percentiles: torch.Tensor) -> None:
        """
        Updates the confusion matrix using the predicted values `predds` and ground truth `labels`.
        0 corresponds to percentile 100

        :param preds: raw predicted values (B)
        :param labels: true values (B)
        :param true_percentiles: the true percentiles in `[0,1]` (B)
        """
        preds = preds.reshape(-1).cpu().detach().clone()

        # save as percentiles
        true_percentiles = true_percentiles.reshape(-1).cpu().detach().clone()
        self.true_percentiles = torch.cat((self.true_percentiles, true_percentiles),
                                          dim=0) if self.true_percentiles is not None else true_percentiles
        # pseudo-percentiles in [0,1] (0 being the lowest and 1 the highest) to percentiles [0,1]
        perc = 1 - preds
        self.perc = torch.cat((self.perc, perc), dim=0) if self.perc is not None else perc

        # CONVERT TO LABELS
        pred_labels = torch.clamp(torch.floor(preds * self.size), min=0, max=self.size - 1)
        super().add(pred_labels, labels)

    @property
    def rmse_percentiles(self):
        return mean_squared_error(y_true=self.true_percentiles, y_pred=self.perc, squared=False)

    @property
    def mae_percentiles(self):
        return mean_absolute_error(y_true=self.true_percentiles, y_pred=self.perc)


def save_pickle(obj, path: Union[str, Path]):
    """
    Saves an object with pickle

    :param obj: object to be saved
    :param save_path: path to the file where it will be saved
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]):
    """
    Loads an object with pickle from a file

    :param path: path to the file where the object is stored
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


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
        save_pickle(d, path)


def load_dict(path: str) -> Dict:
    """
    Loads a dictionary from a file (plain text or pickle)

    :param path: path where the dictionary was saved
    :return: the loaded dictionary
    """
    try:
        return load_pickle(path)
    except pickle.UnpicklingError as e:
        # print(e)
        pass

    with open(path, 'r', encoding="utf-8") as file:
        from ast import literal_eval
        s = file.read()
        return dict(literal_eval(s))


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    :param seed: seed for the random generators
    """
    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)

    # Ensure all operations are deterministic on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # # make deterministic
        # torch.backends.cudnn.determinstic = True
        # torch.backends.cudnn.benchmark = False

        # # for deterministic behavior on cuda >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # set seed for dataset
    ContagionDataset.seed = seed


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

# if __name__ == '__main__':
#     pass
