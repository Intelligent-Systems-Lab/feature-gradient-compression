import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import pandas as pd
import random
import numpy as np
import sys, os
sys.setrecursionlimit(1000000)

def get_optimizer(type, model, lr):
    if type == 'sgd':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)



def get_criterion(device):
    if device == "GPU":
        return nn.NLLLoss().cuda()
    else:
        return nn.NLLLoss()


class MNISTDataset(Dataset):
    """EMNIST dataset"""

    def __init__(self, feature, target, transform=None):

        self.X = []
        self.Y = target

        if transform is not None:
            for i in range(len(feature)):
                self.X.append(transform(feature[i]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.Y[idx]


def getdataloader(dset='./mnist_test.csv', batch=10):
    dset = "/mountdata/{}/{}_train_<ID>.csv".format(dset, dset).replace("<ID>", os.getenv("ID"))
    print("Dataset at : {}".format(dset))

    if dset[-4:] == ".csv":
        train = pd.read_csv(dset)
    elif dset[-2:] == ".p":
        train = pd.read_pickle(dset)

    train = train.values.tolist()
    random.shuffle(train)

    train = train[:int(0.8*len(train))]

    train_labels = np.array([i[0] for i in train])
    train_data = np.array([i[1:] for i in train])
    train_data = train_data.reshape(-1, 28, 28)

    featuresTrain = torch.from_numpy(train_data)
    targetsTrain = torch.from_numpy(train_labels)

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_set = MNISTDataset(featuresTrain.float(), targetsTrain, transform=data_transform)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=2)
    return trainloader


# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
#                                  nn.ReLU(inplace=True),
#                                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(kernel_size=2),
#                                  nn.Dropout(0.25),
#                                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#                                  nn.ReLU(inplace=True),
#                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
#                                  nn.ReLU(inplace=True),
#                                  nn.MaxPool2d(kernel_size=2, stride=2),
#                                  nn.Dropout(0.25))

#         self.classifier = nn.Sequential(nn.Linear(576, 256),
#                                         nn.Dropout(0.5),
#                                         nn.Linear(256, 47))

#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(x.size(0), -1)  # flatten layer
#         x = self.classifier(x)

#         return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
