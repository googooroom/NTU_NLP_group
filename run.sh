#!/bin/bash

# set variables
train_file="/notebooks/nlp/data/train.csv"
# test_file="/notebooks/nlp/data/test.csv"
instruction_file="/notebooks/nlp/data/instructions.txt"
max_len=128
batch_size=32
lr=0.01
epochs=20
weight_decay=1e-4
test_file="/notebooks/nlp/data/test.csv"

# run the main script
python main.py \
    --train_path $train_file \
    --max_len $max_len \
    --batch_size $batch_size \
    --epochs $epochs \
    --lr $lr \
    --weight_decay $weight_decay \
