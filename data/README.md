# How to create dataset for BCFL

## Femnint

Make a dataset from [leaf](https://github.com/TalwalkarLab/leaf.git).

and creat a folder ```femnist```, then copy ```/train``` and ```/test``` in it.

```bash=
python femnist_create.py --data $(pwd)/femnist --n 100
```

## Mnist

```bash=
# set how may package of dataset you want to average packaging.
NUMOFPKG=4 bash create_mnist_dataset.sh

# Result 
# mnist
# |- mnist-in-csv.zip
# |- mnist_train.csv
# |- mnist_test.csv
# |- mnist_train_0.csv
# |- mnist_train_1.csv
# |- .....
```

## Emnist

Emnist contain full hand written image, more detail in [The EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)


### Short describe
* EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
* EMNIST ByMerge: 814,255 characters. 47 unbalanced classes. 👈
* EMNIST Balanced:  131,600 characters. 47 balanced classes.
* EMNIST Letters: 145,600 characters. 26 balanced classes.
* EMNIST Digits: 280,000 characters. 10 balanced classes.
* EMNIST MNIST: 70,000 characters. 10 balanced classes.

![iamge](images/emnist.png))

We use EMNIST By_Merge Dataset

```bash=
# set how may package of dataset you want to average packaging.
NUMOFPKG=4 bash create_emnist_dataset.sh

# Result 
# emnist
# |- emnist-bymerge-mapping.txt
# |- test/
# |- train/
# |- emnist_train_0.csv
# |- emnist_train_1.csv
# |- .....
```

