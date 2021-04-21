import base64
import io

import argparse
import pickle

from options import Configer
from models.models_select import *

torch.nn.Module.dump_patches = True


def object_serialize(d):
    bd = pickle.dumps(d)
    return base64.b64encode(bd).decode()

def object_deserialize(s):
    p = base64.b64decode(s)
    return pickle.loads(p)

def fullmodel2base64(model):
    buffer = io.BytesIO()
    torch.save(model, buffer)
    bg = buffer.getvalue()
    return base64.b64encode(bg).decode()


def base642fullmodel(modbase64):
    inputrpc = bytes(modbase64.encode())
    inputrpc_ = base64.b64decode(inputrpc)
    loadmodel = torch.load(io.BytesIO(inputrpc_))
    return loadmodel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default=None, help='config')
    args = parser.parse_args()

    if args.config is None:
        exit("No config.ini found.")

    con = Configer(args.config)

    if con.trainer.get_dataset() == "mnist":
        Model = Model_mnist
    elif con.trainer.get_dataset() == "mnist_fedavg":
        Model = Model_mnist_fedavg
    elif con.trainer.get_dataset() == "femnist":
        Model = Model_femnist
        # Model = resnet18
    elif con.trainer.get_dataset() == "cifar10":
        Model = ResNet18_cifar

    # with open("/root/FIRSTMOD.txt", "w") as file:
    #     file.write(fullmodel2base64(Model()))
    print(fullmodel2base64(Model()))
