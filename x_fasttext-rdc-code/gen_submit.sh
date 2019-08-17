#! /bin/bash


export PATH=/home/fisher/fastText:$PATH

sh classification_train_model.sh sigir_out/train.out sigir_out/train.out.test sg_model_80_word  sigir_out/

fasttext predict-prob sigir_out/sg_model_80_word.bin sigir_out/test_pre.out 10 > sigir_out/pre_result.out

cat sigir_out/pre_result.out |cut -d ' ' -f1 |sed 's/__label__//g' > sigir_out/test_pre_cate.txt
paste -d '\t' sigir_data/rdc-catalog-test.tsv sigir_out/test_pre_cate.txt > sigir_submit/rdc-catalog-test.tsv
cat sigir_out/pre_result.out|awk '{if($2 >=0.9) print $o}'|cut -f1|wc -l
