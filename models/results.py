from pathlib import Path
from typing import List, Dict
from matplotlib import pyplot as plt
from torchmetrics.functional import auroc

import numpy as np
import pandas as pd

from models.models import load_model, save_model


def confidence_interval(x, significance: float = 5):
    # Calculate the significance % confidence interval for each value
    half_significance = significance / 2
    return np.percentile(x, [half_significance, 100-half_significance]), x.mean()


def plot_bootstrap_graph(x, aucrocs, conf_intervals, label, log_scale=False, x_label='Percentages', y_label='MCC'):
    # Plot the confidence interval for each value
    x = np.log10(x) if log_scale else x
    plt.errorbar(x, aucrocs, yerr=[
        [auc - conf_int[0] for auc, conf_int in zip(aucrocs, conf_intervals)], [conf_int[1] - auc for auc, conf_int in zip(aucrocs, conf_intervals)],
    ], fmt='o-', label=label, capsize=3)
    
    # plt.errorbar(x, [np.mean(auc) for auc in aucrocs], yerr=[
    #     [np.mean(auc) - conf_int[0] for auc, conf_int in zip(aucrocs, conf_intervals)], [conf_int[1] - np.mean(auc) for auc, conf_int in zip(aucrocs, conf_intervals)],
    # ], fmt='o-', label=label)
    
    plt.xticks(x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


class ResultCollection:
    class Result:
        # list of metrics to be filtered
        METRIC_PREFIX = [j + i for i in [
                "mcc",
                "acc",
                "rmse",
                "mae",
                "rmse_perc",
                "mae_perc",
            ] for j in [
                'train_',
                'val_',
                'test_',
            ]
        ]

        def __init__(self, data: List[Dict], uid: str, group: str = None, **kwargs):
            """
            Object used to present the results from the testing

            :param List[Dict] data: list of dictionaries with the different metrics obtained
            :param str uid: unique identifier for the data
            :param str group: group to which the data belongs, defaults to None
            :param kwargs: any other info to attach to the data
            """
            self.uid = uid
            self.data = data
            self.group = group if group is not None else uid
            self.other = kwargs

            self.filter_columns = self.METRIC_PREFIX + ['uid', 'group'] + list(self.other.keys())

        def sort_best(self, metric: str, max: bool = True):
            """
            Sorts the data with respect to the given metric

            :param metric: metric for sorting
            :param max: whether to return the max first
            :return: the sorted data
            """
            sort_idx = np.argsort([k['dict'][metric] for k in self.data])
            if max:
                sort_idx = sort_idx[::-1]
            return [self.data[k] for k in sort_idx]

        def save_best(self, metric: str, folder: str, maximize: bool = True, filename: str = None):
            """
            Saves the best model with a given name

            :param metric: metric for sorting
            :param folder: folder where the model will be saved
            :param maximize: whether to return the max first
            :param filename: filename of the model to be saved
            :return: the best sample
            """
            if filename is None:
                filename = f"best_{self.uid}"

            d = self.sort_best(metric=metric, max=maximize)[0]
            model, d_model = load_model(Path(f"{d['dict']['path_name']}"))
            save_model(model, folder, filename, d_model)

            return d

        @property
        def df(self) -> pd.DataFrame:
            """
            Returns a DataFrame of the data
            """
            return pd.DataFrame(data=[k['dict'] for k in self.data]) \
                .assign(group=self.group, uid=self.uid, **self.other)

        @property
        def df_metrics(self) -> pd.DataFrame:
            """
            Returns a DataFrame of the data metrics
            """
            df = self.df
            cols = [k for k in self.filter_columns if k in df.columns]
            # cols = [k for k in df.columns if any([i in k for i in self.METRIC_PREFIX + ['name']])]
            return df[cols]

        def df_metrics_sort(self, metric: str, maximize: bool = True) -> pd.DataFrame:
            """
            Returns a DataFrame of the data metrics sorted by the given `metric`

            :param metric: metric for sorting
            :param maximize: whether to return the max first
            :return: a pandas DataFrame
            """
            df = self.df_metrics
            return df.sort_values(axis=0, by=metric, ascending=not maximize, na_position='last', ignore_index=True)

    def __init__(self):
        self.results = {}

    def add(self, data: List[Dict], uid: str, group: str = None, **kwargs) -> Result:
        """
        Adds a result to the collection and returns it
        """
        r = self.Result(data, uid, group, **kwargs)
        self.results[uid] = r
        return r

    def df(self, metric: str, maximize: bool = True):
        """
        Returns a DataFrame of the data metrics sorted by the given `metric`.
        The best model (according to `metric`) for each result set is presented

        :param metric: metric for sorting
        :param maximize: whether to return the max first
        :return: a pandas DataFrame
        """
        if not self.results:
            return pd.DataFrame()

        data = [k.df_metrics_sort(metric=metric, maximize=maximize).head(1) for k in self.results.values()]
        df = pd.concat(data)
        df.set_index('uid', inplace=True)
        return df.sort_values(axis=0, by=['group', metric], ascending=not maximize, na_position='last')

# def pretty(ld, indent=0):
#     return None
#     with open('result.txt', 'w', encoding='utf-8') as file:
#         for d in tqdm(ld):
#             file.write('{' + '\n')
#             for key, value in d.items():
#                 file.write('\t' * (indent+1) + str(key) + ':' + str(value) + '\n')
#                 # file.write('\t' * (indent+1) + str(key) + '\n')
#                 # file.write('\t' * (indent+2) + str(value) + '\n')
#             file.write('},\n')
