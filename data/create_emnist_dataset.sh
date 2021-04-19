
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist.py

if [ -z "$NUMOFPKG" ]
then
    NUMOFPKG=5
fi

mkdir emnist
cd emnist
wget https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
unzip gzip.zip
cp gzip/emnist-bymerge* .
rm -r gzip
mkdir {train,test}
gunzip -d emnist-bymerge-test-*
gunzip -d emnist-bymerge-train-*
mv emnist-bymerge-train-* ./train
mv emnist-bymerge-test-* ./test

# pip3 install -r requirements-emnist.txt

cd ..
python3 emnist.py --data $(pwd)/emnist --n $NUMOFPKG