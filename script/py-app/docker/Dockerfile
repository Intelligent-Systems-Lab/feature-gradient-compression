FROM floydhub/pytorch:1.6.0-gpu.cuda10cudnn7-py3.56

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 684BA42D &&\
    curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

RUN apt-get update && apt-get -y upgrade &&\
    pip3 install --upgrade pip setuptools &&\
    pip3 install protobuf==3.14.0 &&\
    pip3 install grpcio &&\
    pip3 install grpcio-tools &&\
    pip3 install pandas &&\
    pip3 install gdown &&\
    pip3 install ipfshttpclient==0.7.0a1

RUN wget https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl &&\
    pip install torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl &&\
    rm torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl &&\
    pip3 install torchvision==0.4.0

RUN gdown --id 15k8E-XuvcatKP1uFqv4pn3PignaKNMOv -O ./abci-0.6.1.tar.gz &&\
    tar zxvf abci-0.6.1.tar.gz &&\
    cd abci-0.6.1 &&\
    python setup.py install &&\
    cd .. &&\
    rm -r abci-0.6.1 abci-0.6.1.tar.gz

RUN add-apt-repository ppa:longsleep/golang-backports &&\
    apt-get update &&\
    apt-get install -y golang-go &&\
    apt-get install -y autotools-dev automake curl git


RUN echo export GOPATH=\"\$HOME/go\" >> ~/.bash_profile &&\
    echo export PATH=\"\$PATH:\$GOPATH/bin\" >> ~/.bash_profile &&\
    echo export GO111MODULE=on >> ~/.bash_profile &&\
    echo "export PATH=\$PATH:/go/bin:/usr/local/go/bin" >>  /root/.bashrc &&\
    mkdir -p $GOPATH/src/github.com/tendermint &&\
    cd $GOPATH/src/github.com/tendermint  &&\
    git clone -b v0.34.0 --single-branch https://github.com/tendermint/tendermint.git &&\
    cd tendermint &&\
    make tools &&\
    make build &&\
    cp build/tendermint /usr/local/bin