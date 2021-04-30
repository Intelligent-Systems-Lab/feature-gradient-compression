import glob
from shutil import copyfile

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
from torch.utils.tensorboard import SummaryWriter
from utils import *
# from models.eminst_model import *
from models.models_select import *
import random
from dgc.warmup import warmup
from db import ipfs

gpu_count = torch.cuda.device_count()


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


def read_state(file):
    j = None
    while 1:
        try:
            with open(file, "r") as f:
                j = json.load(f)
            break
        except json.decoder.JSONDecodeError:
            time.sleep(0.5)
            continue
    return j


def write_state(file, round_, cid, addr):
    s = read_state(file)
    if str(s["data"][-1]["round"]) == str(round_):
        new_incom = {
            "cid": cid,
            "gradient": addr
        }
        s["data"][-1]["incoming_gradient"].append(new_incom)

    with open(file, "w") as f:
        json.dump(s, f, indent=4)


class trainer:
    def __init__(self, config, dataloader, dbHandler, device=torch.device("cpu"), cid=-1, board_path=None):
        self.config = config
        self.cid = cid
        self.dataloader = copy.deepcopy(dataloader)  # path to dataset
        self.dbHandler = dbHandler
        self.device = device
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
        if not board_path is None:
            self.writer = SummaryWriter(board_path)

    def train_run(self, round_, base_model):
        lr = self.warmup.get_lr_from_step(round_)

        Model = get_model(self.config.trainer.get_dataset())

        if type(base_model) is str:
            model = base642fullmodel(self.dbHandler.cat(base_model))
        else:
            model = copy.deepcopy(base_model)

        opt = get_optimizer(self.config.trainer.get_optimizer())
        optimizer = opt(params=model.parameters(),
                        lr=lr,
                        compress_ratio=self.compress_ratio,
                        fusing_ratio=self.fusing_ratio,
                        weight_decay=1e-4,
                        momentum=0.9,
                        nesterov=True,
                        device=self.device)
        model.train().to(self.device)

        eploss = []
        print("train start, {}".format(time.time()))
        for i in range(self.local_ep):
            optimizer.memory.clean()
            print("CID: {}, ep :{}".format(round_, i))

            losses = []
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                # data = data.view(data.size(0),-1)

                output = model(data.float())

                loss = self.loss_function(output, target)
                #print(loss.item())
                losses.append(loss.item())

                loss.backward()

                optimizer.gradient_collect()
                optimizer.step()

            losses = sum(losses) / len(losses)
            eploss.append(losses)
            # optimizer.compress(compress=False)
            # local_g.append(optimizer.decompress(optimizer.get_compressed_gradient()))
        eploss = sum(eploss) / len(eploss)
        self.writer.add_scalar("loss of {}".format(cid), eploss, global_step=round_, walltime=None)
        print("train done, {}".format(time.time()))
        # if self.device == "GPU":
        #     model.cpu()

        # optimizer.memory.clean()
        # for i in local_g:
        #     optimizer.memory.mem.append(i)
        if self.config.trainer.get_optimizer() == "FGCSGD" and round_ > self.config.trainer.get_base_step():
            optimizer.compress(global_momentum=self.cg, compress=True, momentum_correction=self.momentum_correction)
        else:
            optimizer.compress(compress=True, momentum_correction=self.momentum_correction)
        cg = optimizer.get_compressed_gradient()
        time.sleep(random.randint(0, 5))
        print("CID:{} train upload, {}".format(round_, time.time()))
        dbres = self.dbHandler.add(object_serialize(cg))
        print("CID:{} train upload, {}".format(round_, dbres))
        return dbres

    def opt_step_base_model(self, round_, base_model, base_gradient):
        Model = get_model(self.config.trainer.get_dataset())

        model = copy.deepcopy(base_model)

        lr = self.warmup.get_lr_from_step(round_)

        model.to(self.device).train()
        opt = get_optimizer(self.config.trainer.get_optimizer())
        optimizer = opt(params=model.parameters(),
                        lr=lr,
                        weight_decay=1e-4,
                        device=self.device)
        # print("base_grad: {}".format(type(object_deserialize(self.dbHandler.cat(base_gradient)))))
        self.cg = optimizer.decompress(object_deserialize(self.dbHandler.cat(base_gradient)))
        # print("cg: {}".format(type(cg)))
        optimizer.set_gradient(self.cg)
        optimizer.step()
        return copy.deepcopy(model.cpu())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="path to config", type=str, default=None)
    args = parser.parse_args()

    if args.config is None:
        print("Please set --config & --output.")
        exit()

    con_path = os.path.abspath(args.config)
    cid = os.getenv("cid")
    workspace = os.path.abspath(os.getenv("workspace"))
    print("cid : {}".format(cid))
    print("workspace : {}".format(workspace))
    print("GPU : {}".format(int(cid) % gpu_count))

    con_path = os.path.join(os.path.dirname(con_path), "config_run.ini")
    print("Read config: {}".format(con_path))
    config = Configer(con_path)

    dbHandler = ipfs(addr=config.eval.get_ipfsaddr())

    if config.trainer.get_dataset() == "femnist":
        dset = os.path.join(os.path.dirname(workspace), "data")
        dset = os.path.join(dset, config.trainer.get_dataset_path(),
                            "{}_train_{}.csv".format(config.trainer.get_dataset(), cid))
        dloader = getdataloader(dset, batch=config.trainer.get_local_bs())
    elif config.trainer.get_dataset() == "cifar10":
        path = os.path.join(os.path.dirname(workspace), "data", config.trainer.get_dataset_path(),
                            "cifar10_{}.pkl".format(cid))
        dloader = get_cifar_dataloader(root=path, batch=config.trainer.get_local_bs())

    t = trainer(config=config,
                dataloader=dloader,
                cid=cid,
                dbHandler=dbHandler,
                device=torch.device("cuda:{}".format(int(cid) % gpu_count)),
                board_path=os.path.join(workspace, "tfboard"))

    store_base_model = None

    state_file = os.path.join(workspace, "state.json")

    while not os.path.isfile(state_file):
        time.sleep(3)
        print("Wait for init... >> state.json")
        continue

    while not (os.path.isfile(os.path.join(workspace, "models", "training_down"))):
        print("...")
        time.sleep(5)
        state = read_state(state_file)
        # print(state)
        if len(state["data"]) == 0:
            continue

        last_state = state["data"][-1]
        last_incoming = state["data"][-1]["incoming_gradient"]

        if cid in [i["cid"] for i in last_incoming]:
            continue

        if last_state["round"] == 0:
            Model = get_model(config.trainer.get_dataset())
            model = base642fullmodel(dbHandler.cat(last_state["base_result"]))
            store_base_model = copy.deepcopy(model)
        else:
            Model = get_model(config.trainer.get_dataset())
            new_model = t.opt_step_base_model(round_=last_state["round"] - 1,
                                              base_model=store_base_model,
                                              base_gradient=last_state["agg_gradient"])
            store_base_model = copy.deepcopy(new_model)

        if (last_state["round"] % config.bcfl.get_nodes()) == int(cid):
            save_path = os.path.join(workspace, "models", "round_{}.pt".format(last_state["round"]))
            torch.save(store_base_model.state_dict(), save_path)

        if len(state["data"]) >= config.trainer.get_max_iteration():
            continue

        addr = t.train_run(round_=last_state["round"], base_model=store_base_model)
        write_state(file=state_file, round_=last_state["round"], cid=cid, addr=addr)

    # print("Saving...")
    #
    # for i in range(config.trainer.get_max_iteration()):
    #     if (i % config.bcfl.get_nodes()) == int(cid):
    #         save_path = os.path.join(workspace, "models", "round_{}.pt".format(i))
    #         torch.save(store_base_model[i].state_dict(), save_path)

    open(os.path.join(workspace, "models", "save_down"), "w").close()

    print("Exit.")
    exit()

