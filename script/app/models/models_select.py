import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset, sampler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
import pandas as pd
import random
import numpy as np
import sys, os, json

sys.setrecursionlimit(1000000)

# from models.resnet import resnet18
from dgc.optimizer import DGCSGD
from fgc.optimizer import FGCSGD


# def get_optimizer(type_, model, lr, compress_ratio=None, fusing_ratio=None):
def get_optimizer(type_):
    if type_ == 'sgd':
        return torch.optim.SGD  # (model.parameters(), lr=lr, momentum=0.5)
    elif type_ == 'adam':
        return torch.optim.Adam  # (model.parameters(), lr=lr, weight_decay=1e-4)
    elif type_ == 'rms':
        return torch.optim.RMSprop  # (model.parameters(), lr=lr)
    elif type_ == 'DGCSGD':
        return DGCSGD  # (model.parameters(), lr=lr, compress_ratio=compress_ratio)
    elif type_ == 'FGCSGD':
        return FGCSGD  # (model.parameters(), lr=lr, compress_ratio=compress_ratio, fusing_ratio=fusing_ratio)


def get_criterion(type_, device):
    if type_ == "crossentropy":
        c = nn.CrossEntropyLoss()
    elif type_ == "nllloss":
        c = nn.NLLLoss().cuda()

    if device == "GPU":
        return c.cuda()
    else:
        return c


def get_model(type_):
    if type_ == "mnist":
        Model = Model_mnist
    elif type_ == "mnist_fedavg":
        Model = Model_mnist_fedavg
    elif type_ == "femnist":
        Model = Model_femnist
    elif type_ == "cifar10":
        Model = ResNet18_cifar
        # Model = Net
    return Model


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
    # dset = "/mountdata/{}/{}_train_<ID>.csv".format(dset, dset).replace("<ID>", os.getenv("ID"))
    print("Dataset at : {}".format(dset))

    if dset[-4:] == ".csv":
        train = pd.read_csv(dset)
    elif dset[-2:] == ".p":
        train = pd.read_pickle(dset)

    train = train.values.tolist()
    random.shuffle(train)

    # train = train[:int(0.8*len(train))]

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


def get_cifar_dataloader(root='./index.json', client=0, batch=10):
    print("Dataset at : {}".format(root))
    with open(root, 'rb') as fo:
        d = json.load(fo)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if client == -1:
        # test dataset
        cifar_dataset = torchvision.datasets.CIFAR10(root=os.path.dirname(os.path.dirname(root)), train=False,
                                                     download=False, transform=transform)
        cifar_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch,
                                                   num_workers=2)
    else:
        # train dataset
        d = d[str(client)]
        cifar_dataset = torchvision.datasets.CIFAR10(root=os.path.dirname(os.path.dirname(root)), train=True,
                                                     download=False, transform=transform)
        cifar_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch,
                                                   sampler=SubsetRandomSampler(d),
                                                   num_workers=2)
    return cifar_loader


class Model_mnist(nn.Module):
    def __init__(self):
        super(Model_mnist, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Dropout(0.25),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.Dropout(0.25))
        self.classifier = nn.Sequential(nn.Linear(576, 256),
                                        nn.Dropout(0.5),
                                        nn.Linear(256, 10))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # flatten layer
        x = self.classifier(x)
        return x


class Model_mnist_fedavg(nn.Module):
    def __init__(self):
        super(Model_mnist_fedavg, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# def Model_femnist():
#     model = vgg13()
#     model.classifier._modules['3'] = nn.Linear(4096, 512)
#     model.classifier._modules['6'] = nn.Linear(512, 62)
#     model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     return model

class Model_femnist(nn.Module):
    def __init__(self):
        super(Model_femnist, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Dropout(0.25),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.Dropout(0.25))
        self.classifier = nn.Sequential(nn.Linear(576, 256),
                                        nn.Dropout(0.5),
                                        nn.Linear(256, 62))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # flatten layer
        x = self.classifier(x)
        return x


# class Model_cifar(torchvision.models.resnet18)):
#     def __init__(self, ):
#         super(Model_cifar, self).__init__()

class ResNet18_cifar(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet18_cifar, self).__init__(block=torchvision.models.resnet.BasicBlock, layers=[2, 2, 2, 2],
                                             num_classes=10)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
