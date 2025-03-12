# DKR
Code for paper "A Soft-Masking Continual Pre-training Method Based on Domain Knowledge Relevance"
# Requirements  
> * Requires transformers==4.17.0 and Python 3.8
> * See requirements.txt
~~~
conda create --name DKR --file requirements.txt
~~~
# Dataset
The dataset in this paper includes 6 pre-training datasets without annotated corpora and 6 corresponding end-task datasets. Here are the statistical information for each domain:  
|Dataset|Size|Task|#Training|#Testing|#Classes|
|:---|:---|:---|:---|:---|:---|
|Yelp Restaurant|758MB|Aspect Sentiment Classification (ASC)|3452|1120|3|
|Amazon Phone|724MB|Aspect Sentiment Classification (ASC)|239|553|2|
|Amazon Camera|319MB|Aspect Sentiment Classification (ASC)|626|2|
|ACL Papers|867MB|Citation Intent Classification|1520|421|6|
|AI Papers|507MB|Relation Classification|2260|2388|7|
|PubMed Papers|989MB|Chemical-protein Interaction Prediction|2667|7398|13|  
# Continual Pre-training 
Let LM learn a series of domain knowledge through the following:
~~~
max_samples=640000
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
~~~
# End-task Fine-tuning  
After conitinual learning of LM, now we are able to evaluate the performace by runing end-task fine-tuning individually.
~~~
max_samples=640000
seed=(2021 111 222 333 444)
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
~~~
