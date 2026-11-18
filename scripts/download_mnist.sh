#!/bin/sh
# download the 4 MNIST files into data/ (about 11 MB zipped)
# run from the project folder: sh scripts/download_mnist.sh
mkdir -p data
cd data
for name in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    curl -O https://ossci-datasets.s3.amazonaws.com/mnist/$name.gz
    gunzip -f $name.gz
done
