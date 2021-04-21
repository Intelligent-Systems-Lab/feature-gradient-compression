from multiprocessing import Pool
import time, json, requests
import sys, os, base64
import argparse
import subprocess

sys.path.append('script/app')
from script.app.options import Configer
from script.app.models.models_select import *
from script.app.utils import *
from script.app.db import ipfs
from script.app.trainer import trainer
from script.app.aggregator import aggregator
import glob
from tqdm import tqdm
from shutil import copyfile
from script.app.state_controller import State_controller


#############################################
def run_ipfs(logto=None, env=None):
    if logto is None:
        log = subprocess.PIPE
    else:
        log = open(logto, 'a')
    proc = subprocess.Popen('bash ./ipfs/ipfsentrypoint.sh', shell=True, stdout=log, env=env)
    return proc


def run_network_capture(logto=None, env=None):
    if logto is None:
        log = subprocess.PIPE
    else:
        log = open(logto, 'a')
    proc = subprocess.Popen('bash ./network/networkentrypoint.sh', shell=True, stdout=log, stderr=log, env=env)
    return proc


#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="path to config", type=str, default=None)
    parser.add_argument('--output', help="output", type=str, default=None)
    args = parser.parse_args()

    if args.config is None or args.output is None:
        print("Please set --config & --output.")
        exit()

    con_path = os.path.abspath(args.config)
    out_path = os.path.abspath(args.output)

    os.makedirs(out_path, exist_ok=True)

    train_tmp = os.path.abspath("./train_tmp")
    os.makedirs(train_tmp, exist_ok=True)
    procs = []

    # set up config file
    print("Read config: {}".format(con_path))
    copyfile(con_path, os.path.join(os.path.dirname(con_path), "config_run.ini"))
    con_path = os.path.join(os.path.dirname(con_path), "config_run.ini")
    config = Configer(con_path)

    # launch ipfs
    print("Init ipfs.")
    ipfs_log = os.path.join(train_tmp, "ipfs.log")
    p_ipfs = run_ipfs(ipfs_log)
    procs.append(p_ipfs)
    # p_ipfs.wait()

    # init
    states = []
    dbHandler = ipfs(addr=config.eval.get_ipfsaddr())
    Model = get_model(config.trainer.get_dataset())
    state_controller = State_controller()
    trainers = []
    for i in range(config.bcfl.get_nodes()):
        path = os.path.join(os.path.abspath("./"), "data", config.trainer.get_dataset_path(), "index.json")
        trainers.append(trainer(config=config,
                                dataloader=get_cifar_dataloader(root=path, client=i, batch=10),
                                dbHandler=dbHandler))

    aggregator = aggregator(config=config,
                            dbHandler=dbHandler)

    # set env
    env = {
        **os.environ,
        "workspace": str(train_tmp),
    }


    # launch network capture
    print("Init network.")
    # network_log = os.path.join(train_tmp, "network.log")
    # open(network_log, 'a').close()
    # p_network = run_network_capture(logto=network_log, env=env)
    # # p_network.wait()
    # procs.append(p_network)

    time.sleep(10)

    ####################################################################################
    # init first base_model
    first_model = dbHandler.add(object_serialize(Model()))
    time.sleep(1)

    state_controller.add_new_state(round_=0,
                                   agg_gradient="",
                                   base_result=object_deserialize(dbHandler.cat(first_model)))

    for d in range(config.bcfl.get_nodes() - 1):
        time.sleep(1)
        _ = dbHandler.cat(first_model)  # download

    for i in range(config.trainer.get_max_iteration()):
        for j in range(config.bcfl.get_nodes()):
            print("Round :{}, CID :{}".format(i, j))
            update_addr = trainers[j].train_run(round_=i,
                                                base_model=state_controller.get_last_base_model())
            state_controller.add_update(round_=i,
                                        cid=j,
                                        gradient=update_addr)

        agg_addr = aggregator.aggergate_run(round_=i,
                                            gradients=state_controller.get_incoming_gradient())

        for d in range(config.bcfl.get_nodes() - 1):
            time.sleep(1)
            _ = dbHandler.cat(agg_addr)  # download

        new_base_model = trainers[0].opt_step_base_model(round_=i,
                                                         base_model=state_controller.get_last_base_model(),
                                                         base_gradient=agg_addr)

        state_controller.add_new_state(round_=i + 1,
                                       agg_gradient=agg_addr,
                                       base_result=new_base_model)

    # make sure close all process
    for p in p_network:
        p.wait()

    print("Done.\n")
