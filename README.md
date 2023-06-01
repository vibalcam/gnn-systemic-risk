# Predicting Systemic Risk in Financial Systems Using Deep Graph Learning

Code for systemic risk classification and percentile prediction using Graph Neural Networks and Class to Regression (C2R).

Repo for the paper: 

**“Predicting Systemic Risk in Financial Systems Using Deep Graph Learning”**

## Installation

1. Clone this repository (Python 3.9)

    ```shell
    git clone https://github.com/vibalcam/gnn-systemic-risk.git
    ```

2. Install [Pytorch](https://pytorch.org/) and [DGL library](https://www.dgl.ai/).

3. Install Python dependencies 

    ```shell
    pip install -r requirements.txt
    ```

## Code Organization

#### Results 

Inside the `notebooks` folder:
- `results.ipynb` contains the code to summarize the results
- `graphs` folder contains visualizations for the networks
- `results_cm` folder contains the confusion matrices for the classification models
- `results_conf` folder contains visualizations for the confidence intervals

#### Notebooks

Each network has a folder with the different scenarios, the best model for each type, and a notebook (`models_training.ipynb`) with the models' training.

#### Data generation

The code that generates the networks from the aggregated data can be found in each of the network's folders (inside the `notebooks` folder) by the name `generate_data.R`. Each folder also contains a `generate_data.RData` file with the saved workspace.

The networks themselves can be found inside the `data` folder. Each network is divided into two files. The file `network.csv` contains the adjacency matrix of the network, and `nodes.csv` contains the nodes features.

#### Models

The code for the models and training can be found in the [models](models) folder. The `models.py` file contains the model definitions, `train.py` contains the code to train the classification models, and `train_reg.py` contains the code to train the percentile regression models.
