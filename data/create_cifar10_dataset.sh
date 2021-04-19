mkdir -p cifar10/{iid,niid}

cd cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar zxvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz


# rm -r cifar10