from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from models.models import load_model, save_model
import itertools


class ResultCollection:
    class Result:
        # METRIC_PREFIX = ['train_', 'val_', 'test_']
        METRIC_PREFIX = [i+j for i in [
                'train_', 
                'val_', 
                'test_',
            ] for j in [
                "rmse",
                "mae",
                "mcc",
                "acc",
                "rmse_perc",
                "mae_perc",
            ]
        ]

        def __init__(self, data: List[Dict], name: str):
            self.data = data
            self.name = name

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

        def save_best(self, metric: str,folder: str, maximize: bool = True, filename: str = None):
            """
            Saves the best model with a given name

            :param metric: metric for sorting
            :param maximize: whether to return the max first
            :return: the best sample
            """
            if filename is None:
                filename = f"best_{self.name}"

            d = self.sort_best(metric=metric, max=maximize)[0]
            model, d_model = load_model(Path(f"{folder}/{d['dict']['path_name']}"))
            save_model(model, folder, filename, d_model)

            return d

        @property
        def df(self) -> pd.DataFrame:
            """
            Returns a DataFrame of the data
            """
            return pd.DataFrame(data=[k['dict'] for k in self.data]).assign(name=self.name)

        @property
        def df_metrics(self) -> pd.DataFrame:
            """
            Returns a DataFrame of the data metrics
            """
            df = self.df
            cols = [k for k in df.columns if any([i in k for i in self.METRIC_PREFIX + ['name']])]
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

    def add(self, data: List[Dict], name: str) -> Result:
        r = self.Result(data, name)
        self.results[name] = r
        return r

    def df(self, metric: str, maximize: bool = True):
        """
        Returns a DataFrame of the data metrics sorted by the given `metric`.
        The best model (according to `metric`) for each result set is presented

        :param metric: metric for sorting
        :param maximize: whether to return the max first
        :return: a pandas DataFrame
        """
        data = [k.df_metrics_sort(metric=metric, maximize=maximize).head(1) for k in self.results.values()]
        df = pd.concat(data)
        df.set_index('name', inplace=True)
        return df.sort_values(axis=0, by=metric, ascending=not maximize, na_position='last')


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