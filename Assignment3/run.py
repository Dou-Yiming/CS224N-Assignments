#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 3
run.py: Run the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
"""
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from torch._C import device
from datetime import datetime

import os
import pickle
import math
import time
import logging
from tensorboardX import SummaryWriter


from tqdm import tqdm

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------
# Primary Functions
# -----------------


def get_logger(cur_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(os.path.join(cur_path, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(os.path.join(cur_path, 'tb'))

    return logger, writer


def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=5e-4, writer=None):
    """ Train the neural dependency parser.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param output_path (str): Path to which model weights and results are written.
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """
    best_dev_UAS = 0

    # YOUR CODE HERE (~2-7 lines)
    # TODO:
    # 1) Construct Adam Optimizer in variable `optimizer`
    # 2) Construct the Cross Entropy Loss Function in variable `loss_func`
    ###
    # Hint: Use `parser.model.parameters()` to pass optimizer
    # necessary parameters to tune.
    # Please see the following docs for support:
    # Adam Optimizer: https://pytorch.org/docs/stable/optim.html
    # Cross Entropy Loss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)
    # optimizer = optim.SGD(parser.model.parameters(), lr=lr,
    #                       momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=231, T_mult=2, eta_min=1e-5)
    loss_func = nn.CrossEntropyLoss()
    # END YOUR CODE
    step=0
    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS,step = train_for_epoch(
            parser, train_data, dev_data, optimizer, scheduler, loss_func, batch_size, writer,step)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, scheduler, loss_func, batch_size, writer,step):
    """ Train the neural dependency parser for single epoch.

    Note: In PyTorch we can signify train versus test and automatically have
    the Dropout Layer applied and removed, accordingly, by specifying
    whether we are training, `model.train()`, or evaluating, `model.eval()`

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @param batch_size (int): batch size
    @param lr (float): learning rate

    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    parser.model.train()  # Places model in "train" mode, i.e. apply dropout layer
    parser.model = parser.model.to(device)
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            step += 1
            optimizer.zero_grad()   # remove any baggage in the optimizer
            loss = 0.  # store loss for this batch here
            train_x = torch.from_numpy(train_x).long().to(device)
            train_y = torch.from_numpy(train_y.nonzero()[1]).long().to(device)

            # YOUR CODE HERE (~5-10 lines)
            # TODO:
            # 1) Run train_x forward through model to produce `logits`
            # 2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
            # This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
            # between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
            # are the predictions (y^ from the PDF).
            # 3) Backprop losses
            # 4) Take step with the optimizer
            # Please see the following docs for support:
            # Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step
            logits = parser.model(train_x)
            loss = loss_func(logits, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # END YOUR CODE
            writer.add_scalar('loss', float(loss), step)
            writer.add_scalar('learning_rate', float(
                optimizer.state_dict()['param_groups'][0]['lr']), step)
            prog.update(1)
            loss_meter.update(loss.item())

    print("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    parser.model.eval()  # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    writer.add_scalar('dev_UAS', dev_UAS, step)
    return dev_UAS,step


if __name__ == "__main__":
    # Note: Set debug to False, when training on entire corpus
    # debug = True
    debug = False

    cur_path = os.path.join(os.getcwd(), 'exp', time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
    assert not os.path.exists(cur_path), 'Duplicate exp name'
    os.mkdir(cur_path)
    logger, writer = get_logger(cur_path=cur_path)

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(
        debug)

    start = time.time()
    model = ParserModel(embeddings)
    parser.model = model.to(device)
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train(parser, train_data, dev_data, output_path,
          batch_size=8192, n_epochs=10, lr=1e-3, writer=writer)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
