# py abci-app

## install abci lib
```bash=
gdown --id 15k8E-XuvcatKP1uFqv4pn3PignaKNMOv -O ./abci-0.6.1.tar.gz
tar zxvf abci-0.6.1.tar.gz

cd abci-0.6.1
python setup.py build
python setup.py install
cd ..

rm -r abci-0.6.1 abci-0.6.1.tar.gz
```

## docker-compose run
```bash=
# CPU:
docker-compose -f ./docker-compose-py.yml up ipfsA node0 node1 node2 node3

docker-compose -f ./docker-compose-py.yml down -v

# GPU
docker-compose -f ./docker-compose-pygpu.yml up ipfsA node0 node1 node2 node3

docker-compose -f ./docker-compose-pygpu.yml down -v
```


## Send create-task TX
```bash=
# create new model
docker run --rm -it -v $(pwd)/script:/root/:z tony92151/py-abci python3 /root/py-app/utils.py -config /root/py-app/config/config.ini > FIRSTMOD.txt
# sudo chown $(whoami)  FIRSTMOD.txt

# Upload to ipfs
IPFSMOD=$(ipfs --api /ip4/0.0.0.0/tcp/5001 add ./FIRSTMOD.txt -q)
echo $IPFSMOD

# Encode TX into base64 
TX=$(python3 -c "import base64,sys; print(base64.b64encode(sys.argv[1].encode('UTF-8')).decode('UTF-8'))" "{\"type\": \"create_task\",\"max_iteration\": 100,\"sample\": 0.5,\"weight\":\"$IPFSMOD\"}")
echo $TX

curl --header "Content-Type: application/json" -X POST --data "{\"jsonrpc\":\"2.0\", \"method\": \"broadcast_tx_sync\", \"params\": [\"$TX\"], \"id\": 1}" localhost:26657
```

## Eval 

```bash=
cd <path to repo>
docker run --gpus all --rm -it -v $(pwd)/script/py-app:/root/:z -v $(pwd)/data:/mountdata/ tony92151/py-abci python3 /root/eval/eval.py -config /root/config/config.ini
```