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


class aggregator:
    def __init__(self, config, dbHandler):
        self.config = config
        self.dbHandler = dbHandler

    def aggergate_run(self, round_, gradients):
        compressor = DGCCompressor()

        gradient_list = []
        for m in gradients:
            mem = compressor.decompress(object_deserialize(self.dbHandler.cat(m)))
            gradient_list.append(copy.deepcopy(mem))

        agg_gradient = []
        for i in range(len(gradient_list[0])):
            result = torch.stack([j[i] for j in gradient_list]).sum(dim=0)
            agg_gradient.append(result / len(gradient_list))

        # agg_gradient = compressor.memory.avg_mem(mem = gradient_list)

        new_gradient = compressor.compress(agg_gradient, compress=False)

        dbres = self.dbHandler.add(object_serialize(new_gradient))
        return dbres

