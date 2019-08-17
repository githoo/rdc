#! /bin/bash
python pre_deal_data.py --input_file sigir_data/sg_init.txt.random --output_file sigir_out/train.out

head -n 720000 sigir_out/train.out > sigir_out/train.out.train
tail -n 80000 sigir_out/train.out > sigir_out/train.out.test

