FILE=${workspace}/network_tmp
mkdir -p $FILE
touch $FILE/network.pcap
echo "Capture path : $FILE"
python3 ./network/network_capture.py --path $FILE
