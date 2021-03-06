from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import time, json, requests
import sys, os, base64, copy
import argparse
import subprocess
import signal
import torch

sys.path.append('script/app')
from script.app.options import Configer
from script.app.models.models_select import *
from script.app.utils import *
from script.app.db import ipfs
from script.app.trainer import trainer, read_state
from script.app.aggregator import aggregator
from script.app.thread_handler import Treading
import glob
from tqdm import tqdm
from shutil import copyfile, move
from script.app.state_controller import State_controller, state

gpu_count = torch.cuda.device_count()
from getpass import getpass
password = getpass("Please enter your password: ")

def handler(sig, frame, ):
    global procs
    for p in procs:
        p.kill()
    print('Kill all.')
    exit()


signal.signal(signal.SIGINT, handler)


#############################################
def run_ipfs(logto=None, env=None):
    if logto is None:
        log = subprocess.PIPE
    else:
        log = open(logto, 'a')
    proc = subprocess.Popen('bash ./ipfs/ipfsentrypoint.sh', shell=True, stdout=log, stderr=log, env=env)
    return proc


def run_network_capture(logto=None, env=None):
    if logto is None:
        log = subprocess.PIPE
    else:
        log = open(logto, 'a')
    proc = subprocess.Popen('sudo workspace={} bash ./network/networkentrypoint.sh'.format(env["workspace"]), shell=True, stdout=log, stderr=log, stdin=subprocess.PIPE, env=env)
    proc.stdin.write(password.encode())
    proc.stdin.flush()
    return proc

def touch_file(file, root=False, password=None):
    if not root:
        proc = subprocess.Popen("touch {}".format(file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    else:
        proc = subprocess.Popen("sudo touch {}".format(file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        proc.stdin.write(password.encode())
        proc.stdin.flush()
    return proc


def run_trainer(logto=None, env=None):
    if logto is None:
        log = subprocess.PIPE
    else:
        log = open(logto, 'a')
    proc = subprocess.Popen('bash ./script/entrypoint-py.sh', shell=True, stdout=log, stderr=log, env=env)
    return proc


def run_eval(logto=None, env=None):
    if logto is None:
        log = subprocess.PIPE
    else:
        log = open(logto, 'a')
    proc = subprocess.Popen('bash ./script/entrypoint-eval.sh', shell=True, stdout=log, stderr=log, env=env)
    return proc



#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="path to config", type=str, default=None)
    parser.add_argument('--output', help="output", type=str, default=None)
    parser.add_argument('--proc', help="pron", type=str, default=2)
    args = parser.parse_args()

    if args.config is None or args.output is None:
        print("Please set --config & --output.")
        exit()

    con_path = os.path.abspath(args.config)
    out_path = os.path.abspath(args.output)

    # mkdirs
    os.makedirs(out_path, exist_ok=True)

    train_tmp = os.path.abspath("./train_tmp")
    os.makedirs(train_tmp, exist_ok=True)

    models_tmp = os.path.join(train_tmp, "models")
    os.makedirs(models_tmp, exist_ok=True)

    logs_tmp = os.path.join(train_tmp, "logs")
    os.makedirs(logs_tmp, exist_ok=True)

    # set up config file
    print("Read config: {}".format(con_path))
    copyfile(con_path, os.path.join(os.path.dirname(con_path), "config_run.ini"))
    con_path = os.path.join(os.path.dirname(con_path), "config_run.ini")
    config = Configer(con_path)

    procs = []

    # set env
    env = {
        **os.environ,
        "workspace": str(train_tmp),
        "IPFS_PATH": str(os.path.join(train_tmp, "ipfs")),
        "CONFIG": str(con_path)
    }

    # launch ipfs
    print("Init ipfs.")
    ipfs_log = os.path.join(train_tmp, "ipfs.log")
    p_ipfs = run_ipfs(logto=ipfs_log, env=env)
    procs.append(p_ipfs)
    time.sleep(3)
    # p_ipfs.wait()

    # init
    states = []
    dbHandler = ipfs(addr=config.eval.get_ipfsaddr())
    Model = get_model(config.trainer.get_dataset())
    state_controller = State_controller()
    # pool = ThreadPool(3)
    # th = Treading(3)
    print("Init trainer.")
    for i in range(config.bcfl.get_nodes()):
        node_log = os.path.join(train_tmp, "logs", "trainer_{}.log".format(i))
        env_node = copy.deepcopy(env)
        env_node["cid"] = str(i)
        env_node["CUDA_VISIBLE_DEVICES"] = str(i % torch.cuda.device_count())
        p_trainer = run_trainer(logto=node_log, env=env_node)
        procs.append(p_trainer)
        time.sleep(0.2)

    print("Init aggregator.")
    aggregator = aggregator(config=config,
                            dbHandler=dbHandler,
                            device=torch.device("cuda:0"))

    # launch network capture
    print("Init network.")
    network_log = os.path.join(train_tmp, "network.log")
    open(network_log, 'a').close()
    env_ = copy.deepcopy(env)
    env_["PASS"] = str(password)
    p_network = run_network_capture(logto=network_log, env=env_)
    # # p_network.wait()
    procs.append(p_network)

    time.sleep(3)

    print("Init eval.")
    p_eval = run_eval(os.path.join(train_tmp, "logs", "eval.log"), env=env)
    procs.append(p_eval)

    time.sleep(3)

    ####################################################################################
    # init first base_model
    first_model = dbHandler.add(fullmodel2base64(Model()))
    time.sleep(1)

    state_data = state(round_=0,
                       agg_gradient="",
                       base_result=first_model)

    state_file = os.path.join(train_tmp, "state.json")
    with open(state_file, "w") as f:
        json.dump({"data": [eval(state_data.json())]}, f, indent=4)
    # state_controller.model_list.append(object_deserialize(dbHandler.cat(first_model)))

    time.sleep(3)

    pbar = tqdm(total=config.trainer.get_max_iteration())

    st = read_state(state_file)
    last_data = st["data"][-1]
    n_round = len(st["data"])

    while not n_round >= config.trainer.get_max_iteration():
        time.sleep(5)

        if len(last_data["incoming_gradient"]) >= config.bcfl.get_nodes():
            gradients = [i["gradient"] for i in last_data["incoming_gradient"]]
            print("aggergate, {}".format(time.time()))
            agg_addr = aggregator.aggergate_run(gradients=gradients)
            print("aggergate done, {}".format(time.time()))
            state_data = state(round_=last_data["round"] + 1,
                               agg_gradient=agg_addr,
                               base_result="")
            #
            st = read_state(state_file)
            st["data"].append(eval(state_data.json()))
            with open(state_file, "w") as f:
                json.dump(st, f, indent=4)
            time.sleep(1)

        st = read_state(state_file)
        last_data = st["data"][-1]
        n_round = len(st["data"])
        pbar.n = n_round
        pbar.refresh()
    pbar.close()

    time.sleep(10)
    open(os.path.join(models_tmp, "training_down"), "w").close()
    time.sleep(10)
    _ = touch_file(file=os.path.join(train_tmp, "network_tmp", "capture_down"), root=True, password=password)
    time.sleep(5)

    while not len(glob.glob(os.path.join(train_tmp, "models", "round_*.pt"))) >= config.trainer.get_max_iteration():
        time.sleep(3)
        print("wait for saving...")

    while not os.path.isfile(os.path.join(train_tmp, "network_tmp", "convert_down")):
        time.sleep(3)
        print("wait for convert...")

    time.sleep(3)


    # move
    move(con_path, train_tmp)
    #move(train_tmp, out_path)

    # make sure close all process
    time.sleep(5)
    print("Kill all.")
    for p in procs:
        p.kill()

    print("Done.\n")
