#!/bin/bash
max_samples=60000

seed=(2021 111 222 333 444 555 666 777 888 999)

for round in 0;
do
  for idrandom in 0
  do
  for pt_task in 0 1
    do
      for ft_task in $(seq 0 ${pt_task});
        do
          CUDA_VISIBLE_DEVICES=3 python finetune.py \
          --max_seq_length 164 \
          --pt_task ${pt_task} \
          --ft_task ${ft_task} \
          --idrandom ${idrandom} \
          --ntasks 6 \
          --max_samples ${max_samples} \
          --seed ${seed[$round]} \
        --baseline 'das'
      done
    done
  done
done  
