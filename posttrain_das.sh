#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o posttrain_procy_qa-%j.out
#SBATCH --gres gpu:2

# export HF_DATASETS_CACHE="/home/zlz/.cache/huggingface/transformers/dataset_cache/"
export TRANSFORMERS_CACHE="/home/zlz/codes/LLMs/ContinualLM-main_kv_grad/roberta-base"
export CUDA_VISIBLE_DEVICES=3
# --max_train_samples 10000 
max_samples=50000
for idrandom in 0
do
  for pt_task in 0 1 2 3 4 5
  do
    python posttrain.py \
    --per_device_train_batch_size 32 \
    --fp16 \
    --max_seq_length 164 \
    --max_samples ${max_samples} \
    --idrandom ${idrandom} \
    --ntasks 6 \
    --pt_task ${pt_task} \
    --ft_task ${pt_task} \
    --baseline 'das' \
    --max_train_samples 1000 \

  done
done