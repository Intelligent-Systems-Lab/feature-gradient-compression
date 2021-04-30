#!/bin/bash
sleep 1

echo "  ___ ___ _      _      _   ___  "
echo " |_ _/ __| |    | |    /_\ | _ ) "
echo "  | |\__ \ |__  | |__ / _ \| _ \ "
echo " |___|___/____| |____/_/ \_\___/ "
echo
echo

#CONFIG=./script/app/config/config.ini
#export workspace=./train_tmp

python3 -u ./script/app/eval2.py --config ${CONFIG}
