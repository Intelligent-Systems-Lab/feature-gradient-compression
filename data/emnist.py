import gzip
from mlxtend.data import loadlocal_mnist
import pandas as pd
import argparse   
import numpy as np
import matplotlib.pyplot as plt
import random
import math



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--n', type=int, default=5)
parser.add_argument('-f')

args = parser.parse_args()

num_package = args.n

xtrain, ytrain = loadlocal_mnist(
    images_path = args.data + '/train/emnist-bymerge-train-images-idx3-ubyte', 
    labels_path = args.data + '/train/emnist-bymerge-train-labels-idx1-ubyte')

xtest, ytest = loadlocal_mnist(
    images_path = args.data + '/test/emnist-bymerge-test-images-idx3-ubyte', 
    labels_path = args.data + '/test/emnist-bymerge-test-labels-idx1-ubyte')

test_data = np.insert(xtest, 0, ytest, axis=1)

train_data = np.insert(xtrain, 0, ytrain, axis=1)

Xf_all_list = []

for i in range(47):
    Xf_all_list.append([])

for i in range(len(xtrain)):
    Xf_all_list[train_data[i][0]].append(train_data[i])


packages = []

for i in range(num_package):
    list_of_containt = []
    for k in range(47):
        Xp = Xf_all_list[k]
        u = math.floor(len(Xp)/num_package)*i
        d = math.floor(len(Xp)/num_package)*(i+1)
        if i == (num_package-1):
            list_of_containt = list_of_containt + Xf_all_list[k][u:]
        else:
            list_of_containt = list_of_containt + Xf_all_list[k][u:d]
    random.shuffle(list_of_containt)
    packages.append(list_of_containt)

print("This takes about {} minutes, please be patient.".format(2.5*num_package))
for p in range(len(packages)):
    df = pd.DataFrame(data=packages[p])
    df = df.rename(columns={0: "label",1:"1x1", 2:"1x2", 3:"1x3"})
    save_path = args.data+"/emnist_train_{}.csv".format(p)
    print("Create : emnist_train_{}.csv".format(p))
    df.to_csv(save_path,mode = 'w', index=False)

df = pd.DataFrame(data=test_data)
df = df.rename(columns={0: "label",1:"1x1", 2:"1x2", 3:"1x3"})
save_path = args.data+"/emnist_test.csv"
print("Create : emnist_test.csv")
df.to_csv(save_path,mode = 'w', index=False)

df = pd.DataFrame(data=train_data)
df = df.rename(columns={0: "label",1:"1x1", 2:"1x2", 3:"1x3"})
save_path = args.data+"/emnist_train_all.csv"
print("Create : emnist_test.csv")
df.to_csv(save_path,mode = 'w', index=False)