#! /bin/bash

export PATH=/home/fisher/fastText:$PATH

sh classification_train_model.sh sigir_out/train.out.train sigir_out/train.out.test sg_model_72_word  sigir_out/
echo 'finish trained'

fasttext predict-prob sigir_out/sg_model_72_word.bin sigir_out/test_pre.out > sigir_out/pre_result.out

cat sigir_out/pre_result.out|awk '{if($2 >=0.9) print $o}'|cut -f1|wc -l
