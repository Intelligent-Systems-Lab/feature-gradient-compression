#!/bin/bash
sleep 8

echo "  ___ ___ _      _      _   ___  "
echo " |_ _/ __| |    | |    /_\ | _ ) "
echo "  | |\__ \ |__  | |__ / _ \| _ \ "
echo " |___|___/____| |____/_/ \_\___/ "
echo
echo

echo
echo "RUN!!!!"
# read -p "Press [Enter] to continue... or [Control + c] to stop..."
sleep 1

if [ "$MODE" != "" ]
then
    echo -e "\n[Mode] $MODE"
    sleep 1
else
    echo -e "\nPlease set env MODE"
    exit
fi
sleep 1
if [ "$ID" = "0" ]
then
    echo "Removing"
    rm -r /tenconfig/mytestnet
    cd /tenconfig
    tendermint testnet
fi

sleep 5

DATAPATH=/mountdata/$DATASET/"$DATASET"_train_$ID.csv

CONFIG=/root/py-app/config/config_run.ini
if [ -f "$CONFIG" ]; then
    echo "config_run.ini exist."
else
    CONFIG=/root/py-app/config/config.ini
fi

if [ "$MODE" = "core" ]
then
    export TMHOME="/tenconfig/mytestnet/node$ID"

    NUMBEROFGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")

    export CUDA_VISIBLE_DEVICES=$(($ID % $NUMBEROFGPU))

    rm -r /root/logs
    mkdir -p /root/logs

    sed -i 's#laddr = "tcp://127.0.0.1:26657"#laddr = "tcp://0.0.0.0:26657"#'  $TMHOME/config/config.toml
    echo "Start"
    python -u /root/py-app/bcfl.py -config $CONFIG &

    tendermint node --home $TMHOME --proxy_app "tcp://localhost:26658"

else 
    sleep 2

    DOCKERINFO=$(curl -s --unix-socket /run/docker.sock http://docker/containers/json)
    HOSTNAME=$(cat /etc/hostname)

    ID=$(python -c "import sys, json; print([i[\"Names\"][0][1:].split(\"_\")[-1] for i in json.loads(sys.argv[1]) if sys.argv[2] in i[\"Id\"]][0])" "$DOCKERINFO" "$HOSTNAME")
    export ID=$(($ID+4))

    NUMBEROFGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")

    export CUDA_VISIBLE_DEVICES=$(($ID % $NUMBEROFGPU))

    TMHOME="/mytestnet/node_$ID"
    mkdir -p $TMHOME

    tendermint init --home $TMHOME
    # rm /root/.tendermint/config/config.toml
    rm $TMHOME/config/genesis.json
    # cp /tenconfig/mytestnet/node0/config/config.toml /root/.tendermint/config
    cp /tenconfig/mytestnet/node0/config/genesis.json $TMHOME/config

    PP=$(grep persistent_peers /tenconfig/mytestnet/node0/config/config.toml -w)

    # NODEID=$(python -c 'import random; print(random.randint(1,100000))')
    sed -i "s#persistent_peers = \"\"#$PP#"  $TMHOME/config/config.toml
    # tendermint node --proxy_app kvstore
    # python /root/py-app/bcfl.py -dataset $DATAPATH 
    # export ID=0

    # Get container name by query from docker socket
    
    # python -c $'import sys, json;j=json.loads(sys.argv[1]);\nfor i in j: \n\tif i["Id"][:12] == sys.argv[2]: \n\t\t\tprint(i["Names"][0][1:].split("_")[-1])' $A 'ef15fce24dad'
    echo "Start app"
    sleep $(python -c 'import random;print(random.randint(0,3))')
    python -u /root/py-app/bcfl.py -config $CONFIG &
    tendermint node --home $TMHOME
    # echo $hostname
fi
