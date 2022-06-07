# Predicting Systemic Risk Using Deep Graph Learning

This repository is the official implementation of **Predicting Systemic Risk Using Deep Graph Learning**, based on [Pytorch](https://pytorch.org/) and [DGL library](https://www.dgl.ai/).

## Overview

Systemic risk analysis proves of utter importance in guaranteeing the stability of the financial system. Current tools based on Network theory measures or traditional Machine Learning can not use all the available information in their analysis.
On the one hand, network measures focus on the network structure, not taking advantage of node or edge feature information. On the other hand, traditional Machine Learning can take advantage of high volumes of data but fails to consider the network structure.

This work proposes the application of Graph Neural Networks (GNNs) for systemic risk analysis and, more generally, to solve supervised and unsupervised tasks in financial networks. In contrast with network metrics and traditional Machine Learning, GNNs are flexible models capable of using both the network structure and the features of the nodes and edges to perform edge-level, node-level, or graph-level tasks on large networks.

To contrast GNNs' effectiveness against traditional Machine Learning techniques, we apply them to a classification task in generated financial networks in which the nodes are labeled by their systemic importance. GNNs, specially GraphSAGE, surpass traditional Machine Learning techniques in dealing with this task, obtaining better results and verifying their superior capabilities to deal with network data (even when we only pre-label 10% of the nodes). More importantly, these techniques are flexible and can take advantage of all the available information to solve node-level, edge-level, or graph-level supervised and unsupervised tasks related to financial networks.

We also present an approach using data labeled into a small number of classes to train models to predict a regression output, while taking into account its expected distribution. We apply this approach to the above-stated problem to obtain each node's systemic importance percentiles, achieving a better approximation of their true relevance. In sum, this approach presents a way of using a simple pre-labeling to obtain granular and more precise predictions. The only requirements are that the input classes have an order and that each class is associated with a subinterval of the output.

## Installation

1. Clone this repository

```shell
git clone https://github.com/vibalcam/systemic-risk-predictor.git
```

3. Install the dependencies (Python 3.9.5)

```shell
pip install -r requirements.txt
```

## Results

The [notebooks](notebooks) folder contains the different models and results (models_training.ipynb) obtained for each scenario.

The results are summarized in [results.ipynb](notebooks/results.ipynb).