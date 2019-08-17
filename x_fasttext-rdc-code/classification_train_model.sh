#!/bin/bash

trainfilename=$1
testfile=$2
model_name=$3
out_dir=$4

export PATH=/home/fisher/fastText:$PATH

fasttext supervised -input "$trainfilename" -output "$out_dir/$model_name" -dim 100 -lr 0.8 -lrUpdateRate 100 -ws 5 \
-wordNgrams 2 -minCount 1 -minCountLabel 1 -neg 5 -t 1e-5 -bucket 10000000 -epoch 30 -thread 32

fasttext test "$out_dir/$model_name.bin" "$testfile" 
fasttext test "$out_dir/$model_name.bin" "$trainfilename"
