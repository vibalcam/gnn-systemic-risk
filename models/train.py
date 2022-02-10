from os import path
from typing import List

import numpy as np
import torch
import torch.utils.tensorboard as tb
from transformers import BertTokenizerFast

from .models import StateActionModel, load_model, save_model, load_model_from_name
from .utils import ContagionDataset, load_data, accuracy, save_dict, load_dict



g = ContagionDataset()[0]
# LOSS
loss = torch.nn.CrossEntropyLoss().to(device)  # for multiclass classification
# OPTIMIZER

features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']

for epoch in range(n_epochs):
    # forward
    logits = model(g, features)
    # compute predictions
    pred = logits.argmax(1)

    # compute loss over the training set
    loss_val = loss(logits[train_mask], labels[train_mask])
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute accuracy on training/validation
    train_acc = accuracy(pred[train_mask], labels[train_mask])
    val_acc = accuracy(pred[val_mask], labels[val_mask])
    











def train(args):
    """
    Method that trains a given model
    :param args: ArgumentParser with args to run the training (goto main to see the options)
    """
    # cpu or gpu used for training if available (gpu much faster)
    device = torch.device('cuda' if torch.cuda.is_available() and not (args.cpu or args.debug) else 'cpu')
    print(device)

    # Number of epoch after which to validate and save model
    steps_validate = 1

    # Hyperparameters

    # learning rates
    lr: int = args.lr
    # optimizer to use for training
    optimizer_name: str = "adamw"  # adam, adamw, sgd
    # number of epochs to train on
    n_epochs: int = args.n_epochs
    # size of batches to use
    batch_size: int = args.batch_size
    # number of workers (processes) to use for data loading
    num_workers: int = 0 if args.debug else args.num_workers
    # dimensions of the model to use (look at model for more detail)
    shared_out_dim:int = 512
    state_layers:List[int] = [255,125]
    action_layers:List[int] = [255,125]
    # output features
    out_features:int = 1
    # scheduler mode to use for the learning rate scheduler
    scheduler_mode:str = 'max_val_acc'  # min_loss, max_acc, max_val_acc

    # Tensorboard
    global_step = 0
    name_model = f"{optimizer_name}/{scheduler_mode}/{batch_size}/{shared_out_dim},{state_layers},{action_layers}/{lr}/2"
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train', name_model), flush_secs=1)
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid', name_model), flush_secs=1)

    # Model
    dict_model = {
        # dictionary with model information
        "name": name_model,
        "shared_out_dim": shared_out_dim,
        "state_layers": state_layers,
        "action_layers": action_layers,
        "out_features": out_features,
        # metrics
        "train_loss": None,
        "train_acc": 0,
        "val_acc": 0,
        "epoch": 0,
    }
    model = StateActionModel(**dict_model).to(device)

    # Loss
    loss = torch.nn.CrossEntropyLoss().to(device)  # for multiclass classification

    # load train and test data
    dataset = ContagionDataset(sets_lengths=(0.8, 0.1, 0.1), seed=123)

    # del model
    # dict_model = {
    #     # dictionary with model information
    #     "name": name_model,
    #     "shared_out_dim": shared_out_dim,
    #     "state_layers": state_layers,
    #     "action_layers": action_layers,
    #     "out_features": out_features,
    #     "bert_name": bert_name,
    # }
    # model = StateActionModel(**dict_model).to(device)

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

    print(f"{args.log_dir}/{name_model}")

    # train mode and freeze bert
    model.train()
    model.freeze_bert(True)
    for epoch in range(n_epochs):
        print(epoch)
        train_loss = []
        train_acc = []

        # Start training
        for state, action, reward in loader_train:
            # To device
            # state, action, reward = state.to(device), action.to(device), reward.to(device)

            # Compute loss and update parameters
            pred = model(state, action)[:,0]
            loss_val = loss(pred, reward)

            # Do back propagation
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Add train loss and accuracy
            train_loss.append(loss_val.cpu().detach().numpy())
            train_acc.append(accuracy(pred, reward))

        # Evaluate the model
        val_acc = []
        model.eval()
        with torch.no_grad():
            for state, action, reward in loader_valid:
                # To device
                # state, action, reward = state.to(device), action.to(device), reward.to(device)
                pred = model(state, action)[:,0]
                val_acc.append(accuracy(pred, reward))

        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        val_acc = np.mean(val_acc)

        # Step the scheduler to change the learning rate
        if scheduler_mode == "min_loss":
            scheduler.step(train_loss)
        elif scheduler_mode == "max_acc":
            scheduler.step(train_acc)
        elif scheduler_mode == "max_val_acc":
            scheduler.step(val_acc)

        global_step += 1
        if train_logger is not None:
            # train_logger.add_text(model, img)
            train_logger.add_scalar('loss', train_loss, global_step=global_step)
            train_logger.add_scalar('acc', train_acc, global_step=global_step)
            valid_logger.add_scalar('acc', val_acc, global_step=global_step)
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                    global_step=global_step)

        # Save the model
        if (epoch % steps_validate == steps_validate - 1) and (val_acc >= dict_model["val_acc"]):
            print(f"Best val acc {epoch}: {val_acc}")
            name_path = name_model.replace('/', '_')
            save_model(model, f"{args.save_path}/{name_path}")
            dict_model["train_loss"] = train_loss
            dict_model["train_acc"] = train_acc
            dict_model["val_acc"] = val_acc
            dict_model["epoch"] = epoch
            save_dict(dict_model, f"{args.save_path}/{name_path}.dict")
            

if __name__ == '__main__':
    from argparse import ArgumentParser

    args_parser = ArgumentParser()

    args_parser.add_argument('--log_dir', default="./models/logs")
    args_parser.add_argument('--data_path', default="./yarnScripts")
    args_parser.add_argument('--save_path', default="./models/saved")
    # args_parser.add_argument('--age_gender', action='store_true')
    args_parser.add_argument('-t', '--test', type=int, default=None,
                             help='the number of test runs that will be averaged to give the test result,'
                                  'if None, training mode')

    # Hyper-parameters
    args_parser.add_argument('-lr', type=float, default=1e-3, help='learning rates')
    # args_parser.add_argument('-law', '--loss_age_weight', nargs='+', type=float, default=[1e-2],
    #                          help='weight for the age loss')
    # args_parser.add_argument('-opt', '--optimizers', type=str, nargs='+', default=["adam"], help='optimizer to use')
    args_parser.add_argument('-n', '--n_epochs', default=20, type=int, help='number of epochs to train on')
    args_parser.add_argument('-b', '--batch_size', default=1, type=int, help='size of batches to use')
    args_parser.add_argument('-w', '--num_workers', default=2, type=int,
                             help='number of workers to use for data loading')
    # args_parser.add_argument('--non_residual', action='store_true',
    #                          help='if present it will not use residual connections')
    # args_parser.add_argument('--non_max_pooling', action='store_true',
    #                          help='if present the model will not use max pooling (stride in convolutions instead)')
    # args_parser.add_argument('--flatten_out_layer', action='store_true',
    #                          help='if present the model will use flatten before the output linear layer '
    #                               'instead of mean pooling')

    args_parser.add_argument('--cpu', action='store_true')
    args_parser.add_argument('-d', '--debug', action='store_true')

    args = args_parser.parse_args()

    if args.test is None:
        train(args)
    # else:
    #     test(args)
