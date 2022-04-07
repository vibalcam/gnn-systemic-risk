import itertools
from os import path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.utils.tensorboard as tb
from tqdm.auto import tqdm

from .models import load_model, save_model, FNN
from .utils import ConfusionMatrix, ContagionDataset


def train(
        model: torch.nn.Module,
        dict_model: Dict,
        dataset_train: ContagionDataset,
        dataset_val: Optional[ContagionDataset] = None,
        log_dir: str = './models/logs',
        save_path: str = './models/saved',
        lr: float = 1e-2,
        optimizer_name: str = "adamw",
        n_epochs: int = 20,
        scheduler_mode: str = 'max_val_acc',
        debug_mode: bool = False,
        steps_save: int = 1,
        use_cpu: bool = False,
        device=None,
        label_smoothing: float = 0.0,
        use_edge_weight: bool = True,
        scheduler_patience: int = 10,
):
    """
    Method that trains a given model

    :param model: model that will be trained
    :param dict_model: dictionary of model parameters
    :param dataset_train: dataset for training
    :param dataset_val: If not none, dataset for validation. Else it will use the same as for training
    :param log_dir: directory where the tensorboard log should be saved
    :param save_path: directory where the model will be saved
    :param lr: learning rate for the training
    :param optimizer_name: optimizer used for training. Can be `adam, adamw, sgd`
    :param n_epochs: number of epochs of training
    :param scheduler_mode: scheduler mode to use for the learning rate scheduler. Can be `min_loss, max_acc, max_val_acc, max_val_mcc`
    :param use_cpu: whether to use the CPU for training
    :param device: if not none, device to use ignoring other parameters. If none, the device will be used depending on `use_cpu` and `debug_mode` parameters
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    :param steps_save: number of epoch after which to validate and save model (if conditions met)
    :param label_smoothing: label smoothing applied to CrossEntropyLoss (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
    :param use_edge_weight: If true, it uses edge weights for training when possible
    :param scheduler_patience: value used as patience for the learning rate scheduler
    """

    # cpu or gpu used for training if available (gpu much faster)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
    # print(device)

    # Tensorboard
    global_step = 0
    # dictionary of training parameters
    dict_param = {f"tr_par_{k}": v for k, v in locals().items() if k in [
        'lr',
        'optimizer_name',
        'batch_size',
        'scheduler_mode',
        'label_smoothing',
        'use_edge_weight',
        'scheduler_patience',
    ]}
    dict_param.update(dict(
        train_self_loop=dataset_train.add_self_loop,
        train_drop_edges=dataset_train.drop_edges,
    ))
    # dictionary to set model name
    name_dict = dict_model.copy()
    name_dict.update(dict_param)
    # model name
    name_model = '/'.join([
        str(name_dict)[1:-1].replace(',', '/').replace("'", '').replace(' ', '').replace(':', '='),
    ])

    # train_logger = tb.SummaryWriter(path.join(f'{log_dir}_{type(model)}', 'train', name_model), flush_secs=1)
    # valid_logger = tb.SummaryWriter(path.join(f'{log_dir}_{type(model)}', 'valid', name_model), flush_secs=1)
    train_logger = tb.SummaryWriter(path.join(log_dir, str(type(model).__name__), name_model), flush_secs=1)
    valid_logger = train_logger

    # Model
    dict_model.update(dict_param)
    dict_model.update(dict(
        # metrics
        train_loss=None,
        train_acc=0,
        val_acc=0,
        epoch=0,
    ))
    model = model.to(device)

    # Loss
    loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    # datasets
    if dataset_val is None:
        dataset_val = dataset_train

    # load data -> dataset_train given as parameter

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer not configured")

    if scheduler_mode == "min_loss":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience)
    elif scheduler_mode in ["max_acc", "max_val_acc", 'max_val_mcc']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=scheduler_patience)
    else:
        raise Exception("Optimizer not configured")

    # print(f"{name_model}")
    for epoch in range(n_epochs):
        # for epoch in (p_bar := trange(n_epochs, leave = True)):
        # p_bar.set_description(f"{name_model} -> best in {dict_model['epoch']}: {dict_model['val_acc']}")
        # print(f"{epoch} of {n_epochs}")

        train_loss = []
        train_cm = ConfusionMatrix(dataset_train.num_classes, name='train')

        # Start training: train mode
        model.train()
        for g in dataset_train:
            g = g.to(device)

            # Get data
            features = g.ndata['feat'].to(device)
            labels = g.ndata['label'].to(device)
            edge_weight = g.edata['weight'].to(device)
            train_mask = g.ndata['train_mask'].to(device)

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
        val_cm = ConfusionMatrix(dataset_train.num_classes, name='val')

        model.eval()
        with torch.no_grad():
            for g in dataset_val:
                g = g.to(device)

                # Get data
                features = g.ndata['feat']
                labels = g.ndata['label']
                edge_weight = g.edata['weight']
                val_mask = g.ndata['val_mask']

                logits = model(g, features, edge_weight=edge_weight if use_edge_weight else None)

                # Add loss and accuracy
                val_loss.append(loss(logits[val_mask], labels[val_mask]).cpu().detach().numpy())
                val_cm.add(logits[val_mask].argmax(1), labels[val_mask])

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
        elif scheduler_mode == 'max_val_mcc':
            scheduler.step(val_cm.matthews_corrcoef)

        # log metrics
        global_step += 1
        if train_logger is not None:
            # train log
            suffix = 'train'
            train_logger.add_scalar(f'loss_{suffix}', train_loss, global_step=global_step)
            log_confussion_matrix(train_logger, train_cm, global_step, suffix=suffix)
            # validation log
            suffix = 'val'
            valid_logger.add_scalar(f'loss_{suffix}', val_loss, global_step=global_step)
            log_confussion_matrix(valid_logger, val_cm, global_step, suffix=suffix)
            # learning rate log
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # Save the model
        if (reg_save := (epoch % steps_save == steps_save - 1)) or (val_acc >= dict_model["val_acc"]):
            # print(f"Best val acc {epoch}: {val_acc}")
            dict_model["train_loss"] = train_loss
            dict_model["train_acc"] = train_acc
            dict_model["val_acc"] = val_acc
            dict_model["epoch"] = epoch + 1

            name_path = str(list(name_dict.values()))[1:-1].replace(',', '_').replace("'", '').replace(' ', '')
            name_path = f"{dict_model['val_acc']:.2f}_{name_path}"

            # if periodic save, then include epoch
            if reg_save:
                name_path = f"{name_path}_{epoch + 1}"

            save_model(model, save_path, name_path, param_dicts=dict_model)


def log_confussion_matrix(logger, confussion_matrix: ConfusionMatrix, global_step: int, suffix=''):
    """
    Logs the data in the confussion matrix to a logger
    :param logger: tensorboard logger to use for logging
    :param confussion_matrix: confussion matrix from where the metrics are obtained
    :param global_step: global step for the logger
    """
    logger.add_scalar(f'acc_global_{suffix}', confussion_matrix.global_accuracy, global_step=global_step)
    logger.add_scalar(f'acc_avg_{suffix}', confussion_matrix.average_accuracy, global_step=global_step)
    logger.add_scalar(f'mcc_{suffix}', confussion_matrix.matthews_corrcoef, global_step=global_step)
    logger.add_scalar(f'rmse_{suffix}', confussion_matrix.rmse, global_step=global_step)
    for idx, k in enumerate(confussion_matrix.class_accuracy):
        logger.add_scalar(f'acc_class_{idx}_{suffix}', k, global_step=global_step)


def test(
        dataset: ContagionDataset,
        save_path: str = './models/saved',
        n_runs: int = 1,
        debug_mode: bool = False,
        use_cpu: bool = False,
        save: bool = True,
        use_edge_weight: bool = True,
        verbose: bool = False,
) -> Tuple[Dict, float]:
    """
    Calculates the metric on the test set of the model given in args.
    Prints the result and saves it in the dictionary files.

    :param dataset: dataset
    :param save_path: directory where the model will be saved
    :param n_runs: number of runs from which to take the mean
    :param use_cpu: whether to use the CPU for training
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    :param save: whether to save the results in the model dict
    :param use_edge_weight: If true, it uses edge weights for training when possible
    :param verbose: whether to print results

    :return: returns the best model's dict_model, test accuracy and list of all models with test information
    """

    def print_v(s):
        if verbose:
            print(s)

    from pathlib import Path
    # cpu or gpu used for training if available (gpu much faster)
    device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
    print_v(device)
    # # num_workers 0 if debug_mode
    # if debug_mode:
    #     num_workers = 0

    # get model names from folder
    model = None
    best_dict = None
    best_acc = 0.0
    list_all = []
    paths = list(Path(save_path).glob('*'))
    for folder_path in tqdm(paths):
        print_v(f"Testing {folder_path.name}")

        # load model and data loader
        del model
        model, dict_model = load_model(folder_path)
        model = model.to(device).eval()

        # dataset given as parameter

        # start testing
        train_cm = []
        val_cm = []
        test_cm = []
        for k in range(n_runs):
            train_run_cm = ConfusionMatrix(dataset.num_classes, name='train')
            val_run_cm = ConfusionMatrix(dataset.num_classes, name='val')
            test_run_cm = ConfusionMatrix(dataset.num_classes, name='test')

            with torch.no_grad():
                for g in dataset:
                    g = g.to(device)

                    # Get data
                    features = g.ndata['feat']
                    labels = g.ndata['label']
                    edge_weight = g.edata['weight']
                    train_mask = g.ndata['train_mask']
                    val_mask = g.ndata['val_mask']
                    test_mask = g.ndata['test_mask']

                    logits = model(g, features, edge_weight=edge_weight if use_edge_weight else None)

                    train_run_cm.add(logits[train_mask].argmax(1), labels[train_mask])
                    val_run_cm.add(logits[val_mask].argmax(1), labels[val_mask])
                    test_run_cm.add(logits[test_mask].argmax(1), labels[test_mask])

            train_cm.append(train_run_cm)
            val_cm.append(val_run_cm)
            test_cm.append(test_run_cm)

        dict_result = {
            "train_mcc": np.mean([k.matthews_corrcoef for k in train_cm]),
            "val_mcc": np.mean([k.matthews_corrcoef for k in val_cm]),
            "test_mcc": np.mean([k.matthews_corrcoef for k in test_cm]),

            "train_rmse": np.mean([k.rmse for k in train_cm]),
            "val_rmse": np.mean([k.rmse for k in val_cm]),
            "test_rmse": np.mean([k.rmse for k in test_cm]),

            "train_mae": np.mean([k.mae for k in train_cm]),
            "val_mae": np.mean([k.mae for k in val_cm]),
            "test_mae": np.mean([k.mae for k in test_cm]),

            "train_acc": np.mean([k.global_accuracy for k in train_cm]),
            "val_acc": np.mean([k.global_accuracy for k in val_cm]),
            "test_acc": np.mean([k.global_accuracy for k in test_cm]),
        }

        print_v(f"RESULT: {dict_result}")

        dict_model.update(dict_result)
        if save:
            save_model(model, str(folder_path.absolute().parent), folder_path.name, param_dicts=dict_model,
                       save_model=False)

        list_all.append(dict(
            dict=dict_model,
            train_cm=train_cm,
            val_cm=val_cm,
            test_cm=test_cm,
        ))

        # save if best
        if best_acc < (test_acc := dict_model['test_acc']):
            best_acc = test_acc
            best_dict = dict_model

    return best_dict, best_acc, list_all

# if __name__ == '__main__':
#     from argparse import ArgumentParser

#     args_parser = ArgumentParser()

#     args_parser.add_argument('-t', '--test', type=int, default=None,
#                              help='the number of test runs that will be averaged to give the test result,'
#                                   'if None, training mode')

#     args = args_parser.parse_args()

#     if args.test is not None:
#         dataset = ContagionDataset(
#             raw_dir='./data',
#             drop_edges=0,
#             sets_lengths=(0.8, 0.1, 0.1),
#         )
#         test(
#             dataset=dataset,
#             save_path='./notebooks/small_network/saved_fnn',
#             n_runs=1,
#             debug_mode=False,
#             use_cpu=False,
#             save=True,
#             use_edge_weight=True,
#         )
#     else:
#         main_train()
