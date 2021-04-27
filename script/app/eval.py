import argparse
import copy, sys
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from models.models_select import *
from options import Configer
from torch import optim
from tqdm import tqdm
from utils import *


def acc_plot(models, dataloder, config, device="CPU"):
    accd = []

    loss_function = get_criterion(config.trainer.get_lossfun(), device=device)
    
    if not type(dataloder)==list:
        dataloders = []
        for i in range(len(models)):
            dataloders.append(dataloder)
    else:
        dataloders = dataloder

    for i in tqdm(range(len(models))):
        ans = np.array([])
        res = np.array([])
        model = copy.deepcopy(models[i])
        if device == "GPU":
            model.cuda()

        losses = []

        model.eval()
        for data, target in dataloders[i]:
            # data = data.view(data.size(0),-1)
            data = data.float()
            if device == "GPU":
                data = data.cuda()

            output = model(data)

            loss = loss_function(output, target)
            losses.append(loss.item())

            _, preds_tensor = torch.max(output, 1)
            preds = np.squeeze(preds_tensor.cpu().numpy())
            ans = np.append(ans, np.array(target))
            res = np.append(res, np.array(preds))

        losses = sum(losses)/len(losses)
        acc = (ans == res).sum() / len(ans)
        accd.append([acc, losses])

    return accd


def local_training(dataloder, config):
    dataset = config.trainer.get_dataset()
    device = config.trainer.get_device()
    iter_ep = config.trainer.get_max_iteration()
    loacl_ep = config.trainer.get_local_ep()
    lr = config.trainer.get_lr()


    if dataset == "mnist":
        Model = Model_mnist
    elif dataset == "mnist_fedavg":
        Model = Model_mnist_fedavg
    elif dataset == "femnist":
        Model = Model_femnist
    model_ = Model()
    # optimizer = optim.RMSprop(model_.parameters(), lr=0.001)
    # loss_function = nn.CrossEntropyLoss()
    # model_.train()
    models = []

    bmodel = fullmodel2base64(Model())
    for i in tqdm(range(iter_ep*loacl_ep)):
        model = base642fullmodel(bmodel)
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()

        optimizer = get_optimizer(con.trainer.get_optimizer(), model=model, lr=lr)
        loss_function = get_criterion(con.trainer.get_lossfun(), device=device)

        if i % loacl_ep ==0:
            models.append(copy.deepcopy(model).cpu())
        # print("E : ", i)
        running_loss = 0
        for data, target in dataloder:
            if device == "GPU":
                model.train()
                model.cuda()
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            # data = data.view(data.size(0),-1)
            output = model(data.float())
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        bmodel = fullmodel2base64(copy.deepcopy(model).cpu().eval())
        if device == "GPU":
            model_.cpu()
        #models.append(model_.cpu())
        #print(running_loss)
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='config')
    args = parser.parse_args()

    if args.config is None:
        exit("No config.ini found.")

    config = Configer(args.config)
    workspace = os.path.abspath(os.getenv("workspace"))

    reuslt = os.path.join(workspace, "state.json")
    file_ = open(reuslt, 'r')
    context = json.load(file_)
    file_.close()

    Model = get_model(type_=config.trainer.get_dataset())

    print("Prepare test dataloader...")
    if config.trainer.get_dataset() == "femnist":
        dset = os.path.join(os.path.dirname(workspace), "data")
        dset_ = os.path.join(dset, config.trainer.get_dataset_path(), "{}_test.csv".format(config.trainer.get_dataset()))
        test_dataloader = getdataloader(dset_, batch=10)

        # single_test_dataloader = []
        # if con.trainer.get_dataset_path().split("/")[-1] == "niid":
        #     for i in range(con.bcfl.get_scale_nodes() + 4):
        #         dl = getdataloader("/mountdata/{}/single_test/{}_single_{}.csv".format(con.trainer.get_dataset_path(), con.trainer.get_dataset(), i), 10)
        #         single_test_dataloader.append(dl)
    elif config.trainer.get_dataset() == "cifar10":
        path = os.path.join(os.path.dirname(workspace), "data", config.trainer.get_dataset_path(), "cifar10_test.pkl")
        test_dataloader = get_cifar_dataloader(root=path, batch=10)

    bcfl_models = []
    print("Generate acc report...")
    lcontext = []
    context["data"] = context["data"][:config.trainer.get_max_iteration()]
    for i in range(len(context["data"])):
        print("Round: {}".format(context["data"][i]["round"]))
        # time.sleep(0.2)
        base_tmp = Model()
        base_tmp.load_state_dict(torch.load(os.path.join(workspace, "models", "round_{}.pt".format(i))))
        # base_tmp = base642fullmodel(client.cat(context["data"][i]["base_result"]).decode())
        b_acc, loss = acc_plot([base_tmp], test_dataloader, config)[0]
        context["data"][i]["base_acc"] = round(b_acc, 6)
        context["data"][i]["base_loss"] = round(loss, 6)
        
        # This take to long, enable it if you want to know each incoming_model's acc
        if False:
            bases_tmp = []
            dataloaders_tmp = []
            for j in range(len(context["data"][i]["incoming_gradient"])):
                # time.sleep(0.2)
                g_tmp = Model()
                g_tmp.load_state_dict(torch.load("/root/app/save_models/round_{}_cid_{}.pt".format(context["data"][i]["round"], context["data"][i]["incoming_gradient"][j]["cid"])))
                bases_tmp.append(copy.deepcopy(g_tmp))
                if config.trainer.get_dataset_path().split("/")[-1] == "niid":
                    dataloaders_tmp.append(single_test_dataloader[int(context["data"][i]["incoming_gradient"][j]["cid"])])
                else:
                    dataloaders_tmp.append(test_dataloader)

            single_accs = acc_plot(bases_tmp, dataloaders_tmp)

            for j in range(len(context["data"][i]["incoming_gradient"])):
                context["data"][i]["incoming_gradient"][j]["single_acc"] = round(single_accs[j], 6)
    
    with open(os.path.join(workspace, "state_report.json"), 'w') as f:
        json.dump(context, f, indent=4)

    # bcfl_result = acc_plot(bcfl_models, test_dataloader, con.trainer.get_device()).
    bcfl_result = [i["base_acc"] for i in context["data"]]
    bcfl_loss = [i["base_loss"] for i in context["data"]]

    miter = config.trainer.get_max_iteration()

    fig, ax1 = plt.subplots()
    plt.title(config.eval.get_title())
    plt.grid(True)
    ax2 = ax1.twinx()
    ax1.plot(range(miter), bcfl_result[:miter], 'r-')
    ax2.plot(range(miter), bcfl_loss[:miter], 'g-')

    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy', color='r')
    ax2.set_ylabel('loss', color='g')

    plt.savefig(os.path.join(workspace, config.eval.get_output()))
