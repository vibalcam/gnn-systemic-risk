import itertools
from os import path
from typing import List, Dict

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import GCN, GAT, GraphSAGE, load_model, save_model
from .utils import ConfusionMatrix, ContagionDataset, save_dict, load_dict


def train(
        model: torch.nn.Module,
        dict_model: Dict,
        dataset: ContagionDataset,
        log_dir: str = './models/logs',
        save_path: str = './models/saved',
        lr: float = 1e-2,
        optimizer_name: str = "adamw",
        n_epochs: int = 20,
        scheduler_mode: str = 'max_val_acc',
        debug_mode: bool = False,
        steps_validate: int = 1,
        use_cpu: bool = False,
        label_smoothing: float = 0.0,
        use_edge_weight: bool = True,
):
    """
    Method that trains a given model

    :param model: model that will be trained
    :param dict_model: dictionary of model parameters
    :param dataset: dataset
    :param log_dir: directory where the tensorboard log should be saved
    :param save_path: directory where the model will be saved
    :param lr: learning rate for the training
    :param optimizer_name: optimizer used for training. Can be adam, adamw, sgd
    :param n_epochs: number of epochs of training
    :param scheduler_mode: scheduler mode to use for the learning rate scheduler. Can be min_loss, max_acc, max_val_acc
    :param use_cpu: whether to use the CPU for training
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    :param steps_validate: number of epoch after which to validate and save model (if conditions met)
    :param label_smoothing: label smoothing applied to CrossEntropyLoss (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
    :param use_edge_weight: If true, it uses edge weights for training when possible
    """

    # cpu or gpu used for training if available (gpu much faster)
    device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
    print(device)

    # Tensorboard
    global_step = 0
    dict_param = {k: v for k, v in locals().items() if k in [
        'lr',
        'optimizer_name',
        'batch_size',
        'scheduler_mode',
        'label_smoothing',
        'use_edge_weight',
    ]}
    name_model = '/'.join([
        str(dict_model)[1:-1].replace(',', '/').replace("'", '').replace(' ', '').replace(':', '='),
        '/',
        str(dict_param)[1:-1].replace(',', '/').replace("'", '').replace(' ', '').replace(':', '='),
    ])
    train_logger = tb.SummaryWriter(path.join(log_dir, 'train', name_model), flush_secs=1)
    valid_logger = tb.SummaryWriter(path.join(log_dir, 'valid', name_model), flush_secs=1)

    # Model
    dict_model.update(dict(
        name=name_model,
        # metrics
        train_loss=None,
        train_acc=0,
        val_acc=0,
        epoch=0,
    ))
    model = model.to(device)

    # Loss
    loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    # load data
    # loader_train, loader_valid, _ = load_data(
    #     dataset_path=data_path,
    #     num_workers=num_workers,
    #     batch_size=batch_size,
    #     drop_last=False,
    #     random_seed=123,
    #     tokenizer=model.tokenizer,
    #     device=device,
    # )

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer not configured")

    if scheduler_mode == "min_loss":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    elif scheduler_mode in ["max_acc", "max_val_acc"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    else:
        raise Exception("Optimizer not configured")

    print(f"{name_model}")

    for epoch in range(n_epochs):
        print(f"{epoch} of {n_epochs}")
        train_loss = []
        train_cm = ConfusionMatrix(dataset.num_classes)

        # Start training: train mode
        model.train()
        for g in dataset:
            g = g.to(device)

            # Get data
            features = g.ndata['feat']
            labels = g.ndata['label']
            edge_weight = g.edata['weight']
            train_mask = g.ndata['train_mask']

            # Compute loss on training and update parameters
            logits = model(g, features, edge_weight=edge_weight if use_edge_weight else None)
            loss_train = loss(logits[train_mask], labels[train_mask])

            # Do back propagation
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # Add train loss and accuracy
            train_loss.append(loss_train.cpu().detach().numpy())
            train_cm.add(logits[train_mask].argmax(1), labels[train_mask])

        # Evaluate the model
        # Can be done in combination with training if drop_edges is 0
        # No performance issue for small graphs so we can separate it
        val_loss = []
        val_cm = ConfusionMatrix(dataset.num_classes)
        test_cm = ConfusionMatrix(dataset.num_classes)
        model.eval()
        with torch.no_grad():
            for g in dataset:
                g = g.to(device)

                # Get data
                features = g.ndata['feat']
                labels = g.ndata['label']
                edge_weight = g.edata['weight']
                val_mask = g.ndata['val_mask']
                test_mask = g.ndata['test_mask']

                logits = model(g, features, edge_weight=edge_weight if use_edge_weight else None)

                # Add loss and accuracy
                val_loss.append(loss(logits[val_mask], labels[val_mask]).cpu().detach().numpy())
                val_cm.add(logits[val_mask].argmax(1), labels[val_mask])
                test_cm.add(logits[test_mask].argmax(1), labels[test_mask])

        # calculate mean metrics
        train_loss = np.mean(train_loss)
        train_acc = train_cm.global_accuracy
        val_loss = np.mean(val_loss)
        val_acc = val_cm.global_accuracy

        # Step the scheduler to change the learning rate
        if scheduler_mode == "min_loss":
            scheduler.step(train_loss)
        elif scheduler_mode == "max_acc":
            scheduler.step(train_acc)
        elif scheduler_mode == "max_val_acc":
            scheduler.step(val_acc)

        # log metrics
        global_step += 1
        if train_logger is not None:
            # train log
            train_logger.add_scalar('loss', train_loss, global_step=global_step)
            log_confussion_matrix(train_logger, train_cm, global_step)
            # validation log
            valid_logger.add_scalar('loss', val_loss, global_step=global_step)
            log_confussion_matrix(valid_logger, val_cm, global_step)
            # learning rate log
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # Save the model
        if (epoch % steps_validate == steps_validate - 1) and (val_acc >= dict_model["val_acc"]):
            # todo add more info
            print(f"Best val acc {epoch}: {val_acc}")
            dict_model["train_loss"] = train_loss
            dict_model["train_acc"] = train_acc
            dict_model["val_acc"] = val_acc
            dict_model["epoch"] = epoch
            name_path = name_model.replace('/', '_')
            save_model(model, save_path, name_path, param_dicts=dict_model)


def log_confussion_matrix(logger, confussion_matrix: ConfusionMatrix, global_step: int):
    """
    Logs the data in the confussion matrix to a logger
    :param logger: tensorboard logger to use for logging
    :param confussion_matrix: confussion matrix from where the metrics are obtained
    :param global_step: global step for the logger
    """
    logger.add_scalar('acc_global', confussion_matrix.global_accuracy, global_step=global_step)
    logger.add_scalar('acc_avg', confussion_matrix.average_accuracy, global_step=global_step)
    for idx, k in enumerate(confussion_matrix.class_accuracy):
        logger.add_scalar(f'acc_class_{idx}', k, global_step=global_step)


# def test(
#         data_path: str = './yarnScripts',
#         save_path: str = './models/saved',
#         n_runs: int = 1,
#         batch_size: int = 8,
#         num_workers: int = 0,
#         debug_mode: bool = False,
#         use_cpu: bool = False,
#         save: bool = True,
# ) -> None:
#     """
#     Calculates the metric on the test set of the model given in args.
#     Prints the result and saves it in the dictionary files.

#     :param data_path: directory where the data can be found
#     :param save_path: directory where the model will be saved
#     :param n_runs: number of runs from which to take the mean
#     :param batch_size: size of batches to use
#     :param num_workers: number of workers (processes) to use for data loading
#     :param use_cpu: whether to use the CPU for training
#     :param debug_mode: whether to use debug mode (cpu and 0 workers)
#     :param save: whether to save the results in the model dict
#     """
#     from pathlib import Path
#     # cpu or gpu used for training if available (gpu much faster)
#     device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
#     print(device)
#     # num_workers 0 if debug_mode
#     if debug_mode:
#         num_workers = 0

#     # get model names from folder
#     model = None
#     for folder_path in Path(save_path).glob('*'):
#         print(f"Testing {folder_path.name}")

#         # load model and data loader
#         del model
#         model, dict_model = load_model(folder_path)
#         model = model.to(device).eval()
#         _, _, loader_test = load_data(
#             dataset_path=data_path,
#             num_workers=num_workers,
#             batch_size=batch_size,
#             drop_last=False,
#             random_seed=123,
#             tokenizer=model.tokenizer,
#             device=device,
#         )

#         # start testing
#         test_acc = []
#         for k in range(n_runs):
#             run_acc = []

#             with torch.no_grad():
#                 for state, action, reward in loader_test:
#                     pred = model(state, action)[:, 0]
#                     run_acc.append(accuracy(pred, reward))

#             run_acc = np.mean(run_acc)
#             print(f"Run {k}: {run_acc}")
#             test_acc.append(run_acc)

#         test_acc = np.mean(test_acc)
#         dict_result = {"test_acc": test_acc}

#         print(f"{folder_path.name}: {dict_result}")
#         dict_model.update(dict_result)
#         if save:
#             save_dict(dict_model, f"{folder_path}/{folder_path.name}.dict")


def main_train():
    dataset = ContagionDataset(
        raw_dir='./data',
        drop_edges=0,
        sets_lengths=(0.8, 0.1, 0.1),
    )

    gcn_model = dict(
        in_features=[dataset.node_features],
        h_features=[[5, 5], [5, 10], [10, 5]],
        out_features=[dataset.num_classes],
        activation=[torch.nn.ReLU()],
        norm_edges=['both'],
        norm_nodes=[None, 'bn', 'gn'],
        dropout=[0.2, 0.5, 0.0],
        # other
        lr=[1e-2, 1, 1e-3],
        label_smoothing=[0.0, 0.2, 0.4],
    )
    list_gcn_model = [dict(zip(gcn_model.keys(), k)) for k in itertools.product(*gcn_model.values())]

    for d in list_gcn_model:
        lr = d.pop('lr')
        ls = d.pop('label_smoothing')
        train(
            model=GCN(**d),
            dict_model=d,
            dataset=dataset,
            log_dir='./models/logs',
            save_path='./models/saved',
            lr=lr,
            optimizer_name="adamw",
            n_epochs=100,
            scheduler_mode='max_val_acc',
            debug_mode=False,
            steps_validate=1,
            use_cpu=False,
            label_smoothing=ls,
            use_edge_weight=True,
        )


if __name__ == '__main__':
    from argparse import ArgumentParser
    args_parser = ArgumentParser()

    args_parser.add_argument('-t', '--test', type=int, default=None,
                             help='the number of test runs that will be averaged to give the test result,'
                                  'if None, training mode')

    args = args_parser.parse_args()

    if args.test is not None:
        pass
    else:
        main_train()
