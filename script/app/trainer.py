import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim
import argparse
import base64
import io, os, json
import ipfshttpclient
import time
import copy
import thread_handler as th
from messages import AggregateMsg, UpdateMsg

from utils import *
# from models.eminst_model import *
from models.models_select import *
import random
from dgc.warmup import warmup


def train(logger, dbHandler, config, bmodel, _round, sender, dataloader, lr, mome=None):
    local_ep = config.trainer.get_local_ep()
    device = config.trainer.get_device()
    mc = config.dgc.get_momentum_correction()
    cr = config.dgc.get_compress_ratio()
    fr = 0.8
    # lr = config.trainer.get_lr()
    # lr = lr

    if config.trainer.get_dataset() == "mnist":
        Model = Model_mnist
    elif config.trainer.get_dataset() == "mnist_fedavg":
        Model = Model_mnist_fedavg
    elif config.trainer.get_dataset() == "femnist":
        Model = Model_femnist
        # Model = resnet18
    elif config.trainer.get_dataset() == "cifar10":
        Model = ResNet18_cifar
    # model_ = Model()

    model = Model()
    if type(bmodel) == str:
        try:
            model = base642fullmodel(dbHandler.cat(bmodel))
            # logger.info("ipfs success : {}".format(model[:20]))
        except TimeoutError:
            logger.info("ipfs fail")
    else:
        model = copy.deepcopy(bmodel)

    if device == "GPU":
        model.cuda()

    optimizer = get_optimizer(config.trainer.get_optimizer(), model=model, lr=lr, compress_ratio=cr, fusing_ratio=fr)
    loss_function = get_criterion(config.trainer.get_lossfun(), device=device)

    if config.trainer.get_optimizer() == "DGCSGD":
        optimizer.memory.clean()
    elif config.trainer.get_optimizer() == "FGCSGD":
        optimizer.memory.clean()

    model.train()
    # logger.info("Train model dataloader")
    local_g = []

    eploss = []
    for i in range(local_ep):
        if config.trainer.get_optimizer() == "DGCSGD":
            optimizer.memory.clean()
        elif config.trainer.get_optimizer() == "FGCSGD":
            optimizer.memory.clean()

        losses = []
        for data, target in dataloader:
            if device == "GPU":
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            # data = data.view(data.size(0),-1)

            output = model(data.float())

            loss = loss_function(output, target)

            loss.backward()

            optimizer.gradient_collect()
            optimizer.step()

            losses.append(loss.item())
        losses = sum(losses) / len(losses)
        eploss.append(losses)
        # optimizer.compress(compress=False)
        # local_g.append(optimizer.decompress(optimizer.get_compressed_gradient()))
    eploss = sum(eploss) / len(eploss)
    add_value_file(path='/root/app/loss_{}.json'.format(os.getenv("ID")), value=eploss)

    if device == "GPU":
        model.cpu()

    # optimizer.memory.clean()
    # for i in local_g:
    #     optimizer.memory.mem.append(i)
    if config.trainer.get_optimizer() == "FGCSGD" and _round > config.trainer.get_base_step():
        optimizer.compress(global_momentum=mome, compress=True, momentum_correction=mc)
    else:
        optimizer.compress(compress=True, momentum_correction=mc)
    cg = optimizer.get_compressed_gradient()

    dbres = dbHandler.add(object_serialize(cg))
    # UpdateMsg.set_cid(os.getenv("ID"))

    result = UpdateMsg()
    result.set_cid(os.getenv("ID"))
    result.set_round(_round)
    result.set_weight(dbres)
    # result.set_cid(os.getenv("ID"))
    # time.sleep(3)
    send_result = sender.send(result.json_serialize())
    return send_result


def add_value_file(path, value):
    if not os.path.exists(path):
        f = open(path, 'w')
        json.dump({"data": []}, f)
        f.close()

    file_ = open(path, 'r')
    context = json.load(file_)
    file_.close()

    context["data"].append(value)

    file_ = open(path, 'w')
    json.dump(context, file_)
    file_.close()


class trainer:
    def __init__(self, config, dataloader, dbHandler):
        self.config = config
        self.dataloader = dataloader  # path to dataset
        self.dbHandler = dbHandler
        self.devices = self.config.trainer.get_device()
        self.local_bs = self.config.trainer.get_local_bs()
        self.local_ep = self.config.trainer.get_local_ep()
        self.compress_ratio = self.config.dgc.get_compress_ratio()
        self.fusing_ratio = self.config.dgc.get_fusing_ratio()
        self.momentum = self.config.dgc.get_momentum()
        self.momentum_correction = self.config.dgc.get_momentum_correction()

        self.cg = None  # last gradient

        self.warmup = warmup(start_lr=self.config.trainer.get_start_lr(),
                             max_lr=self.config.trainer.get_max_lr(),
                             min_lr=self.config.trainer.get_min_lr(),
                             base_step=self.config.trainer.get_base_step(),
                             end_step=self.config.trainer.get_end_step())

        self.loss_function = get_criterion(self.config.trainer.get_lossfun(), device=self.device)

    def train_run(self, round_, base_model):
        lr = self.warmup.get_lr_from_step(round_)

        Model = get_model(self.config.trainer.get_dataset())

        if type(base_model) is str:
            model = base642fullmodel(self.dbHandler.cat(base_model))
        else:
            model = copy.deepcopy(base_model)

        opt = get_optimizer(self.config.trainer.get_optimizer())
        optimizer = opt(model=model,
                        lr=lr,
                        compress_ratio=self.compress_ratio,
                        fusing_ratio=self.fusing_ratio)
        eploss = []
        for i in range(self.local_ep):
            self.optimizer.memory.clean()

            losses = []
            for data, target in self.dataloader:
                if self.device == "GPU":
                    data = data.cuda()
                    target = target.cuda()

                self.optimizer.zero_grad()
                # data = data.view(data.size(0),-1)

                output = model(data.float())

                loss = self.loss_function(output, target)

                loss.backward()

                optimizer.gradient_collect()
                optimizer.step()

                losses.append(loss.item())
            losses = sum(losses) / len(losses)
            eploss.append(losses)
            # optimizer.compress(compress=False)
            # local_g.append(optimizer.decompress(optimizer.get_compressed_gradient()))
        eploss = sum(eploss) / len(eploss)

        if self.device == "GPU":
            model.cpu()

        # optimizer.memory.clean()
        # for i in local_g:
        #     optimizer.memory.mem.append(i)
        if self.config.trainer.get_optimizer() == "FGCSGD" and round_ > self.config.trainer.get_base_step():
            optimizer.compress(global_momentum=self.cg, compress=True, momentum_correction=self.momentum_correction)
        else:
            optimizer.compress(compress=True, momentum_correction=self.momentum_correction)
        cg = optimizer.get_compressed_gradient()

        dbres = self.dbHandler.add(object_serialize(cg))
        return dbres

    def opt_step_base_model(self, round_, base_model, base_gradient):
        Model = get_model(self.config.trainer.get_dataset())

        model = copy.deepcopy(base_model)

        lr = self.warmup.get_lr_from_step(round_)

        model.cpu().train()
        opt = get_optimizer(self.config.trainer.get_optimizer())
        optimizer = opt(model=model,
                        lr=lr)
        # print("base_grad: {}".format(type(object_deserialize(self.dbHandler.cat(base_gradient)))))
        self.cg = optimizer.decompress(object_deserialize(self.dbHandler.cat(base_gradient)))
        # print("cg: {}".format(type(cg)))
        optimizer.set_gradient(self.cg)
        optimizer.step()
        return copy.deepcopy(model.cpu())

    def get_model_by_ipfs(self, key):
        if self.config.trainer.get_dataset() == "mnist":
            Model = Model_mnist
        elif self.config.trainer.get_dataset() == "mnist_fedavg":
            Model = Model_mnist_fedavg
        elif self.config.trainer.get_dataset() == "femnist":
            Model = Model_femnist
        elif self.config.trainer.get_dataset() == "cifar10":
            Model = ResNet18_cifar
        # model_ = Model()

        model = Model()
        model = base642fullmodel(self.dbHandler.cat(key))
        # txmanager.set_last_base_model(model.cpu())

        return model

    # def trainRun(self, bmodel):
    #     print("Run train")
