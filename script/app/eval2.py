import argparse
import copy, sys
import glob
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from models.models_select import *
from options import Configer
from torch import optim
from tqdm import tqdm
from utils import *
from torch.utils.tensorboard import SummaryWriter

def acc_plot(model, dataloder, config, device="CPU"):
    accd = []

    loss_function = get_criterion(config.trainer.get_lossfun(), device=device)

    if device == "GPU":
        model.cuda()

    losses = []
    ans = np.array([])
    res = np.array([])

    model.eval()
    for data, target in dataloder:
        # data = data.view(data.size(0),-1)
        data = data.float()
        if device == "GPU":
            data = data.cuda()
            target = target.cuda()

        output = model(data)

        loss = loss_function(output, target)
        losses.append(loss.item())

        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        ans = np.append(ans, np.array(target.cpu()))
        res = np.append(res, np.array(preds))

    losses = sum(losses)/len(losses)
    acc = (ans == res).sum() / len(ans)

    return acc, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='config')
    args = parser.parse_args()

    if args.config is None:
        exit("No config.ini found.")

    config = Configer(args.config)
    workspace = os.path.abspath(os.getenv("workspace"))

    Model = get_model(type_=config.trainer.get_dataset())
    writer = SummaryWriter(os.path.join(workspace, "tfboard"))

    print("Prepare test dataloader...")
    if config.trainer.get_dataset() == "femnist":
        dset = os.path.join(os.path.dirname(workspace), "data")
        dset_ = os.path.join(dset, config.trainer.get_dataset_path(), "{}_test.csv".format(config.trainer.get_dataset()))
        test_dataloader = getdataloader(dset_, batch=config.trainer.get_local_bs())
    elif config.trainer.get_dataset() == "cifar10":
        path = os.path.join(os.path.dirname(workspace), "data", config.trainer.get_dataset_path(), "cifar10_test.pkl")
        test_dataloader = get_cifar_dataloader(root=path, batch=config.trainer.get_local_bs())

    last_iter = -1
    while True:
        time.sleep(5)
        print("...")
        l = glob.glob(os.path.join(workspace, "models", "round_*.pt"))
        if len(l)==0:
            continue
        l = [int(i.split("/")[-1].split("_")[1].split(".")[0]) for i in l]

        if last_iter == max(l):
            continue

        last_iter = max(l)
        print("Eval round: {}".format(last_iter))
        base_tmp = Model()
        base_tmp.load_state_dict(torch.load(os.path.join(workspace, "models", "round_{}.pt".format(last_iter))))
        b_acc, loss = acc_plot(base_tmp, test_dataloader, config, "GPU")
        writer.add_scalar("test loss ", loss, global_step=last_iter, walltime=None)
        writer.add_scalar("test acc ", b_acc, global_step=last_iter, walltime=None)

        if last_iter+1 >= config.trainer.get_max_iteration():
            break


