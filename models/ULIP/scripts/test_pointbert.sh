#!/bin/bash

if [ -z "$1" ]; then
echo "Please provide a *.pt file as input"
exit 1
fi

model_file=$1
output_dir=./outputs/test_pointbert_8kpts_modelnet

CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_PointBERT --npoints 8192 --output-dir $output_dir --evaluate_3d --comp_path /ibex/project/c2106/Mahmoud/datasets/shards/test_captions_3p.json  --normal --test_ckpt_addr $model_file 2>&1 | tee $output_dir/log.txt