import pickle
import os, json
import random, copy
import argparse
from shutil import copyfile


def unpickle(files):
    data = []
    data_c = [[] for i in range(10)]
    data_index = [[] for i in range(10)]
    for f in files:
        print("Read data: {}".format(f))
        with open(f, 'rb') as fo:
            d = pickle.load(fo, encoding='latin1')

        for i in range(len(d['labels'])):
            data.append(d['labels'][i])
            data_c[d['labels'][i]].append((d['labels'][i], d['data'][i], d['filenames'][i]))

    for d in range(len(data)):
        data_index[data[d]].append(d)

    return data_c, data_index


def pickling(f, value):
    with open(f, 'wb') as fo:
        pickle.dump(value, fo)


def divide_part(vlaue, parts):
    vlaue = copy.deepcopy(vlaue)
    # random.shuffle(vlaue)
    c = 0
    d = [[] for i in range(parts)]
    for i in vlaue:
        d[c % parts].append(i)
        c += 1
    random.shuffle(d)
    return d


cifar_path = ["cifar-10-batches-py/data_batch_1",
              "cifar-10-batches-py/data_batch_2",
              "cifar-10-batches-py/data_batch_3",
              "cifar-10-batches-py/data_batch_4",
              "cifar-10-batches-py/data_batch_5"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('-f')
    args = parser.parse_args()

    path = os.path.abspath(args.data)
    number_of_client = args.n

    paths = [os.path.join(path, i) for i in cifar_path]
    data, data_index = unpickle(paths)

    os.makedirs(os.path.join(path, "iid"), exist_ok=True)
    os.makedirs(os.path.join(path, "niid"), exist_ok=True)
    # for i in range(number_of_client):
    #     os.makedirs(os.path.join(path, "niid", "cifar10_train_{}".format(i), "cifar-10-batches-py"))
    #     os.makedirs(os.path.join(path, "iid", "cifar10_train_{}".format(i), "cifar-10-batches-py"))   

    k = {'batch_label': '', 'labels': [], 'data': [], 'filenames': []}

    chunks = [[] for i in range(number_of_client)]

    # for d in data_index:
    for d in data:
        dp = divide_part(d, number_of_client)

        for i in range(len(dp)):
            chunks[i] = chunks[i] + dp[i]

    # clients_json = {}
    #
    # for i in range(len(chunks)):
    #     clients_json["{}".format(i)] = chunks[i]
    #
    # with open(os.path.join(path, "iid", "index.json"), 'w') as fo:
    #     json.dump(clients_json, fo)
    for i in range(len(chunks)):
        k_ = copy.deepcopy(k)
        for d in chunks[i]:
            k_["labels"].append(d[0])
            k_["data"].append(d[1])
            k_["filenames"].append(d[2])
        pickling(os.path.join(path, "iid", "cifar10_{}.pkl".format(i)), k_)
        print("Save result: {}".format(os.path.join(path, "iid", "cifar10_{}.pkl".format(i))))

    copyfile(os.path.join(path, "cifar-10-batches-py", "test_batch"), os.path.join(path, "iid", "cifar10_test.pkl"))



