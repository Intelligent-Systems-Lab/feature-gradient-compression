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
from fgc.fgc import FGCCompressor


class aggregator:
    def __init__(self, config, dbHandler, device=torch.device("cpu")):
        self.config = config
        self.dbHandler = dbHandler
        self.device = device

    def aggergate_run(self, gradients):
        compressor = FGCCompressor(device=self.device)

        gradient_list = []
        print("aggergate add , {}".format(time.time()))
        for m in gradients:
            mem = compressor.decompress(object_deserialize(self.dbHandler.cat(m)))
            mem = [i.to(self.device) for i in mem]
            gradient_list.append(copy.deepcopy(mem))

        agg_gradient = []
        for i in range(len(gradient_list[0])):
            result = torch.stack([j[i].to(self.device) for j in gradient_list]).sum(dim=0)
            agg_gradient.append(result / len(gradient_list))

        # agg_gradient = compressor.memory.avg_mem(mem = gradient_list)
        agg_gradient = [i.to(self.device) for i in agg_gradient]
        print("aggergate compress, {}".format(time.time()))
        new_gradient = compressor.compress(agg_gradient, compress=False)

        dbres = self.dbHandler.add(object_serialize(new_gradient))
        return dbres

