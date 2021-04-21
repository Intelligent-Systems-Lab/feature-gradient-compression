FILE=${workspace}/network_tmp
mkdir -p $FILE
echo "Capture path : $FILE"
python3 ./network/network_capture.py --path $FILE
