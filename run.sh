#!/bin/bash

# set variables
train_file="/content/data/train.csv"
# test_file="/notebooks/nlp/data/test.csv"
instruction_file="/content/data/instructions.txt"
max_len=64
batch_size=64
lr=1e-5
epochs=10
weight_decay=1e-4
test_file="/content/data/test.csv"
prompt_length=20
fig_name='roberta-10epoch-base'

# run the main script
# python main.py \
#     --fig_name $fig_name \
#     --train_path $train_file \
#     --max_len $max_len \
#     --batch_size $batch_size \
#     --epochs $epochs \
#     --lr $lr \
#     --weight_decay $weight_decay 

#     # --prompt_length $prompt_length \
#     # --apply_soft_prompt


## baseline
python main.py \
    --fig_name $fig_name \
    --train_path $train_file \
    --max_len $max_len \
    --batch_size $batch_size \
    --baseline