import os
import sys
import struct
import argparse
import pandas as pd
import random
import math
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--n', type=int, default=5)
parser.add_argument('--format', type=str, default='csv', help="[csv, pickle]")
parser.add_argument('-f')

def get_dict_idx():
    d = {0:"label"}
    count = 1
    for i in range(1, 29):
        for j in range(1, 29):
            d[count] = "{}x{}".format(i, j)
            count += 1
    return d

args = parser.parse_args()

X = pd.read_csv(args.data + "/mnist_train.csv")
num_package = args.n
# X_test = pd.read_csv("/Users/tonyguo/Desktop/mnist/mnist-in-csv/mnist_test.csv")

X_df = pd.DataFrame(X)

Xf_all_list = []

train_data = X_df.values.tolist()

for i in range(10):
    Xf_all_list.append([])

for i in tqdm(range(len(train_data))):
    Xf_all_list[train_data[i][0]].append(train_data[i])

packages = []

for i in tqdm(range(num_package)):
    list_of_containt = []
    for k in range(10):
        Xp = Xf_all_list[k]
        u = math.floor(len(Xp) / num_package) * i
        d = math.floor(len(Xp) / num_package) * (i + 1)
        if i == (num_package - 1):
            list_of_containt = list_of_containt + Xf_all_list[k][u:]
        else:
            list_of_containt = list_of_containt + Xf_all_list[k][u:d]
    random.shuffle(list_of_containt)
    packages.append(list_of_containt)

# for p in range(len(packages)):
#     pk = pd.concat(packages[p])
#     pk = pk.sample(frac=1).reset_index(drop=True)
#     del pk["index"]
#     if args.format == "pickle":
#         save_path = args.data+"/mnist_train_{}.p".format(p)
#         print("Create : mnist_train_{}.p".format(p))
#         pk.to_pickle(save_path)
#     else:
#         save_path = args.data+"/mnist_train_{}.csv".format(p)
#         print("Create : mnist_train_{}.csv".format(p))
#         pk.to_csv(save_path,mode = 'w', index=False)

for p in range(len(packages)):
    df = pd.DataFrame(data=packages[p])
    df = df.rename(columns=get_dict_idx())
    if args.format == "pickle":
        save_path = args.data + "/mnist_train_{}.p".format(p)
        print("Create : mnist_train_{}.p".format(p))
        df.to_pickle(save_path)
    else:
        save_path = args.data + "/mnist_train_{}.csv".format(p)
        print("Create : mnist_train_{}.csv".format(p))
        df.to_csv(save_path, mode='w', index=False)



