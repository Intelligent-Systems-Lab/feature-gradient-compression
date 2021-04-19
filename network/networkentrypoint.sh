apk add wireshark tshark
chgrp wireshark /usr/bin/dumpcap

# DATE=$(date "+%H_%M_%S")
# FILE=/root/network/network_tmp/network_$DATE.pcap
FILE=/root/network/network_tmp/network.pcap
mkdir -p /root/network/network_tmp
touch $FILE
echo "Capture file : $FILE"
python3 -u /root/network/network_capture.py $FILE
# tshark -i bcfl -w $FILE
