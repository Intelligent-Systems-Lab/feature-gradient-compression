import os, sys, json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import math
import argparse

def get_dict_idx():
    d = {0:"label"}
    count = 1
    for i in range(1, 29):
        for j in range(1, 29):
            d[count] = "{}x{}".format(i, j)
            count += 1
    return d

def image_invert(img):
    return [int(255-i) for i in img]


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in tqdm(files):
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])
    clients = list(sorted(data.keys()))
    return clients, groups, data

def image_invert(img):
    return [int(255-i) for i in img]

def mix_all_writers(data):
    """
    > input:
    [ [writer1, [[label, ...], [label, ...], [label, ...], ... ]],
      [writer2, [[label, ...], [label, ...], [label, ...], ... ]],
       . 
       .
       .
    ]

    > output:
    [[label, ...], [label, ...], [label, ...], ... ]
    """
    output = []
    for i in range(len(data)):
        for j in data[i][1]:
            output.append(j)
    random.shuffle (output)
    return output

def class_mix_data(data):
    """
    > input:
    [[label, ...], [label, ...], [label, ...], ... ]

    > output:
    [ [[label, ...], [label, ...], [label, ...], ... ],     <- class 0
      [[label, ...], [label, ...], [label, ...], ... ],     <- class 1
       . 
       .
       .
      [[label, ...], [label, ...], [label, ...], ... ],     <- class 61
    ]
    """
    all_list = []
    for i in range(62):
        all_list.append([])
    
    for i in data:
        all_list[i[0]].append(i) # append images by label

    return all_list

def niid_train_test(train_data, test_data, path):
    
    train_sampled_list = train_data
    test_sampled_list = test_data

    # save train non-iid data to csv files
    niid_path = os.path.join(path, "niid")
    print("Save train niid data to : {}".format(niid_path))
    for p in tqdm(range(len(train_sampled_list))):
        df = pd.DataFrame(data=train_sampled_list[p][1])
        df = df.rename(columns=get_dict_idx())
        save_path = os.path.join(niid_path, "femnist_train_{}.csv".format(p))
        df.to_csv(save_path, mode='w', index=False)
    ########################################################################
    # save train non-iid data to csv files
    niid_path = os.path.join(path, "niid", "single_test")
    print("Save test niid single data to : {}".format(niid_path))
    for p in tqdm(range(len(test_sampled_list))):
        df = pd.DataFrame(data=test_sampled_list[p][1])
        df = df.rename(columns=get_dict_idx())
        save_path = os.path.join(niid_path, "femnist_single_{}.csv".format(p))
        df.to_csv(save_path, mode='w', index=False)
    ########################################################################
    mix_test_data = mix_all_writers(test_sampled_list)
    # save test non-iid data to csv files
    niid_path = os.path.join(path, "niid")
    save_path = os.path.join(niid_path, "femnist_test.csv")
    print("Save test niid data to : {}".format(save_path))
    df = pd.DataFrame(data=mix_test_data)
    df = df.rename(columns=get_dict_idx())
    df.to_csv(save_path, mode='w', index=False)


def iid_train_test(train_data, test_data, number_of_client, path):
    mix_train_data = mix_all_writers(train_data)
    class_train_data = class_mix_data(mix_train_data)
    """
    images in each class separated in n chunk

    [ [[[label, ...], [label, ...]], [[label, ...],[label, ...]],  ... ],     <- class 0, 2 images in a chunk
      [[[label, ...], [label, ...]], [[label, ...],[label, ...]],  ... ],     <- class 1, 2 images in a chunk
       . 
       .
       .
      [[[label, ...], [label, ...]], [[label, ...],[label, ...]],  ... ],     <- class 61, 2 images in a chunk
    ]
    """
    chunk_class_train_data = []
    # for i in range(62):
    #     chunk_class_train_data.append([])

    for i in range(len(class_train_data)):
        l = class_train_data[i]

        chunked_list = []
        for i in range(number_of_client):
            chunked_list.append([])
        
        count = 0
        for i in l:
            chunked_list[count].append(i)
            count += 1
            if count >= (number_of_client-1):
                count = 0
            
        random.shuffle(chunked_list)
        chunk_class_train_data.append(chunked_list)
    
    packages = []
    num_package = number_of_client
    for i in range(num_package):
        list_of_containt = []
        for k in range(len(chunk_class_train_data)):
            # print("{},{}".format(i,k))
            # print(chunk_class_train_data[k][i])
            list_of_containt = list_of_containt + chunk_class_train_data[k][i]
        random.shuffle(list_of_containt)
        packages.append(list_of_containt)

    iid_path = os.path.join(path, "iid")
    print("Save train iid data to : {}".format(iid_path))
    for p in tqdm(range(len(packages))):
        df = pd.DataFrame(data=packages[p])
        df = df.rename(columns=get_dict_idx())
        save_path = os.path.join(iid_path, "femnist_train_{}.csv".format(p))
        df.to_csv(save_path, mode='w', index=False)
    ########################################################################
    mix_test_data = mix_all_writers(test_data)
    # save test iid data to csv files
    niid_path = os.path.join(path, "iid")
    save_path = os.path.join(niid_path, "femnist_test.csv")
    print("Save test iid data to : {}".format(save_path))
    df = pd.DataFrame(data=mix_test_data)
    df = df.rename(columns=get_dict_idx())
    df.to_csv(save_path, mode='w', index=False)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--n', type=int, default=100, help="There are 3500 client in this dataset, we will random selected n client of it.")
    parser.add_argument('-f')
    
    args = parser.parse_args()
    
    if args.data is None:
        exit()

    path = args.data
    number_of_client = args.n

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")

    os.mkdir(os.path.join(path, "iid"))
    os.mkdir(os.path.join(path, "niid"))
    os.mkdir(os.path.join(path, "niid","single_test"))
    
    print("Read data from : {}".format(train_path))    
    x_train, f_train, d_train = read_dir(train_path)
    
    print("Read data from : {}".format(test_path)) 
    x_test, f_test, d_test = read_dir(test_path)

    # [ [writer1, [[label, ...], [label, ...], [label, ...], ... ]],
    #   [writer2, [[label, ...], [label, ...], [label, ...], ... ]],
    #    . 
    #    .
    #    .
    # ]

    ########################################################################
    ## sample data from both train and test dataset ########################
    random_sample = random.sample(list(range(len(x_train))), k = number_of_client)
    
    # append smaped writer's name in the list
    train_sampled_list = []
    for i in range(len(random_sample)):
        train_sampled_list.append([x_train[i],[]])

    test_sampled_list = train_sampled_list.copy()
    
    ########################################################################
    ## append train and test image into list  ##############################
    for i in range(len(train_sampled_list)):
        writer = train_sampled_list[i][0]
        images = []
        for j in range(len(d_train[writer]["y"])):
            label = [int(d_train[writer]["y"][j])]
            image = image_invert(d_train[writer]["x"][j])
            # label = [int(i) for i in label]
            img = label+image
            images.append(img)
        train_sampled_list[i][1].extend(images)

    for i in range(len(test_sampled_list)):
        writer = test_sampled_list[i][0]
        images = []
        for j in range(len(d_test[writer]["y"])):
            label = [int(d_test[writer]["y"][j])]
            image = image_invert(d_test[writer]["x"][j])
            # label = [int(label)]
            img = label+image
            images.append(img)
        test_sampled_list[i][1].extend(images)

    # Both train and test data format:
    # [ [writer1, [[label, ...], [label, ...], [label, ...], ... ]],
    #   [writer2, [[label, ...], [label, ...], [label, ...], ... ]],
    #    . 
    #    .
    #    .
    # ]

    print("Train images: {}".format(len(mix_all_writers(train_sampled_list))))
    print("Test images: {}".format(len(mix_all_writers(test_sampled_list))))

    ########################################################################
    ## process niid dataset ################################################
    niid_train_test(train_sampled_list, test_sampled_list, path)

    ########################################################################
    ## process iid dataset #################################################
    iid_train_test(train_sampled_list, test_sampled_list, number_of_client, path)
    
    print("Done")
