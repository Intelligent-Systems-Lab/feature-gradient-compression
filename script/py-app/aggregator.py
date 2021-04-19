import sys

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim
import sys
import time
from concurrent import futures
import logging

import argparse
import base64
import io, os, copy
import ipfshttpclient
import thread_handler as th
from messages import AggregateMsg, UpdateMsg
import random
from utils import *
from models.eminst_model import *
import hashlib

from models.models_select import *
from dgc.dgc import DGCCompressor


def aggergate(logger, dbHandler, gradients, _round, sender, config):
    logger.info("Agg start")
    logger.info("Len of models : {}".format(len(gradients)))
    compressor = DGCCompressor()
    gradient_list = []

    if config.trainer.get_dataset() == "mnist":
        Model = Model_mnist
    elif config.trainer.get_dataset() == "mnist_fedavg":
        Model = Model_mnist_fedavg
    elif config.trainer.get_dataset() == "femnist":
        Model = Model_femnist
        # Model = resnet18
    elif config.trainer.get_dataset() == "cifar10":
        Model = ResNet18_cifar

    for m in gradients:
        mem = compressor.decompress(object_deserialize(dbHandler.cat(m)))
        gradient_list.append(copy.deepcopy(mem))

    agg_gradient = []
    for i in range(len(gradient_list[0])):
        result = torch.stack([j[i] for j in gradient_list]).sum(dim=0)
        agg_gradient.append(result / len(gradient_list))

    # agg_gradient = compressor.memory.avg_mem(mem = gradient_list)

    new_gradient = compressor.compress(agg_gradient, compress=False)


    # new_model_state = gradient_list[0].state_dict()

    # # sum the weight of the model
    # for m in gradient_list[1:]:
    #     state_m = m.state_dict()
    #     for key in state_m:
    #         new_model_state[key] += state_m[key]
    #
    # # average the model weight
    # for key in new_model_state:
    #     new_model_state[key] /= len(model_list)
    #
    # new_model = model_list[0]
    # new_model.load_state_dict(new_model_state)

    # dbres = dbHandler.add(fullmodel2base64(new_model))
    dbres = dbHandler.add(object_serialize(new_gradient))
    #dbres = models[0]

    # AggregateMsg.set_cid(os.getenv("ID"))

    result = AggregateMsg()
    result.set_cid(os.getenv("ID"))
    result.set_round(_round+1)
    result.set_weight(gradients)
    result.set_result(dbres)
    # result.set_cid(os.getenv("ID"))
    time.sleep(random.randint(3, 5))
    send_result = sender.send(result.json_serialize())
    logger.info("Agg done")
    return send_result


class aggregator:
    def __init__(self, logger, config, dbHandler, sender):
        self.logger = logger
        self.config = config
        self.dbHandler = dbHandler
        self.sender = sender
        self.hash = hashlib.md5()

    def aggergate_run(self, bmodels, round_):
        t = th.create_job(aggergate, (self.logger, self.dbHandler, bmodels, round_, self.sender, self.config))
        t.start()
        self.logger.info("Run Agg")

    def aggergate_check(self, modtx) -> bool:
        rou = modtx["Round"]
        wei = modtx["Weight"]
        return True

    def aggergate_manager(self, txmanager, tx):
        if txmanager.aggregation_lock and (not tx["type"] == "aggregate_again"):
            return

        if tx["type"] == "aggregate_again" or (tx["type"] == "update" and len(txmanager.get_incoming_gradient()) >= txmanager.threshold):
            txmanager.aggregation_lock = True
            self.aggregator_selection(txmanager)

            if str(txmanager.get_last_state()["aggregator_id"]) == str(os.getenv("ID")):
                self.logger.info("It's me!!!!!!!!!!!!!!!!")
                self.aggergate_run(txmanager.get_incoming_gradient(), txmanager.get_last_round())
        else:
            return

    def aggregator_selection(self, txmanager):
        self.logger.info(">>>>>>>> nonce :{}".format(txmanager.get_last_state()["selection_nonce"]))
        self.logger.info("hash {} + {}".format(txmanager.get_last_gradient_result(), str(txmanager.get_last_state()["selection_nonce"])))

        self.hash.update((txmanager.get_last_gradient_result() + str(txmanager.get_last_state()["selection_nonce"])).encode())
        tmpsum = 0
        for j in self.hash.hexdigest():
            tmpsum += ord(j)
        txmanager.get_last_state()["aggregator_id"] = tmpsum % txmanager.get_last_state()["number_of_validator"]

        self.logger.info("selection {}".format(txmanager.get_last_state()["aggregator_id"]))

