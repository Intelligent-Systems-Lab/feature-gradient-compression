echo "Copy key..."

mkdir -p /data/ipfs/
cp /root/ipfskey/swarm.key /data/ipfs/ 


echo 'Initialize IPFS ...'

if [ -d "/root/ipfs/" ]; then
    echo "Removing /root/ipfs/..."
  rm -rf /root/.ipfs/
  rm -rf /root/ipfs/
fi

ipfs init
ipfs config Addresses.API /ip4/0.0.0.0/tcp/5001
ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8080
echo 'Removing default bootstrap nodes...'
ipfs bootstrap rm --all

#apk add jq
ipfs id
PEERID=$(ipfs id | jq ."ID")
PEERID="${PEERID:1}"
PEERID="${PEERID::-1}"

if [ ! -f "/ipfscon/ipfsaddr.txt" ]; then
    mkdir /ipfscon
    touch /ipfscon/ipfsaddr.txt
fi
#rm /ipfscon/ipfsaddr.txt
#touch /ipfscon/ipfsaddr.txt

echo "/ip4/$HOSTIP/tcp/4001/ipfs/$PEERID" >> /ipfscon/ipfsaddr.txt

sleep 5

while read line; do 
    #echo $line 
    ipfs bootstrap add $line 
done < /ipfscon/ipfsaddr.txt

ipfs daemon
# ipfs daemon >> /$HOSTIP.log 2>&1 &


