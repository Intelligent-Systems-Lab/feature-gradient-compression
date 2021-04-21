FROM golang:1.14.4-alpine

RUN apk add --no-cache git make bash gcc musl-dev jq

RUN mkdir /installiipfs &&\
    cd /installiipfs &&\
    wget https://github.com/ipfs/go-ipfs/releases/download/v0.7.0/go-ipfs-source.tar.gz &&\
    mkdir -p $GOPATH/src/github.com/ipfs/go-ipfs &&\
    tar zxvf go-ipfs-source.tar.gz -C $GOPATH/src/github.com/ipfs/go-ipfs &&\
    cd $GOPATH/src/github.com/ipfs/go-ipfs &&\
    make install
