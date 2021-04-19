
# https://scidm.nchc.org.tw/dataset/mnist

if [ -z "$NUMOFPKG" ]
then
    NUMOFPKG=5
fi

if [ -z "$FORMAT" ]
then
    FORMAT=pickle
fi

mkdir mnist_fedavg
cd mnist_fedavg

if [ -f "./mnist-in-csv.zip" ]; then
    echo "exist"
    # unzip mnist-in-csv.zip
else 
    wget https://scidm.nchc.org.tw/dataset/mnist/resource/ce2b188c-bb3e-41ab-8dfc-d9b6e665fbd4/nchcproxy -O mnist-in-csv.zip
    unzip mnist-in-csv.zip
fi

cd ..
python3 mnist_fedavg.py --data $(pwd)/mnist_fedavg --n $NUMOFPKG --format $FORMAT
