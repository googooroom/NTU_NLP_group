# NTU_NLP_group
1. chmod +x run.sh
2. edit the arguments in run.sh file 
3. ./run.sh to run the file

In Google Colab:
1. Copy all files to run time
2. Create a folder 'diagram' to save result
3. Run `!bash /content/run.sh` to run the file
4. To activate baseline, add --baseline to the command: 

    ```
    python main.py \
    --fig_name $fig_name \
    --train_path $train_file \
    --max_len $max_len \
    --batch_size $batch_size \
    --baseline
    ```


5. For fine tuning, modify the hyperparameters

    ```
    python main.py \
    --fig_name $fig_name \
    --train_path $train_file \
    --max_len $max_len \
    --batch_size $batch_size \
    --epochs $epochs \
    --lr $lr \
    --weight_decay $weight_decay 
    ```

6. For softprompt tuning, 

    ```
    python main.py \
    --fig_name $fig_name \
    --train_path $train_file \
    --max_len $max_len \
    --batch_size $batch_size \
    --epochs $epochs \
    --lr $lr \
    --weight_decay $weight_decay \
    --prompt_length $prompt_length \
    --apply_soft_prompt
    ```