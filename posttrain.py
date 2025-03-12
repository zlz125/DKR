#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# limitations under the License.

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['https_proxy'] = '127.0.0.1:7890'
# os.environ['http_proxy'] = '127.0.0.1:7890'
import utils
import logging
import random
import torch
import datasets
import transformers
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    AutoConfig,
    RobertaTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
    set_seed,
)
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from config import parseing_posttrain
from dataloader.data import get_dataset
from torch.utils.data import DataLoader
from utils.data import group_texts
from datasets import Dataset, DatasetDict, concatenate_datasets
import torch.nn as nn
import finetune
import finetune_add_auc

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def main():
    args = parseing_posttrain()
    # args = parseing_common()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # prepare_sequence_posttrain为增量预训练的领域顺序准备
    args = utils.model.prepare_sequence_posttrain(args)
    from approaches.posttrain import Appr
    from finetune_add_auc import Acc

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(fp16=args.fp16, kwargs_handlers=[ddp_kwargs])
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            pass
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    print('*' * 20 + args.model_name_or_path + "*" * 20)
    # tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(
        "/home/zlz/codes/LLMs/ContinualLM-main_kv_grad/roberta-base")
    args.tokenizer = tokenizer

    model = utils.model.lookfor_model_posttrain(args)
    # param=model.model.roberta.state_dict()
    # for n, p in accelerator.unwrap_model(model).named_parameters():
    #     # if p.grad is not None:
    #     print(f'Gradient of param "{n}" with size {tuple(p.size())} detected')

    # print(model.state_dict().keys())
    accelerator.wait_for_everyone()

    # print(model)

    if 'comb' in args.baseline:
        for t in range(args.pt_task + 1):
            if t == 0:
                raw_datasets = get_dataset(args.data[t], tokenizer=None, args=args)

            else:
                cur_raw_datasets = get_dataset(args.data[t], tokenizer=None, args=args)
                train_dataset = cur_raw_datasets["train"]

                raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], train_dataset])
    else:
        # Get the dataset
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = get_dataset(args.dataset_name, tokenizer=None, args=args)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # # 另外处理新的评估子数据集
    # seed = 42
    print("训练集长度" + str(len(raw_datasets["train"])))
    ori_train_datasets = raw_datasets["train"]
    nsamples = 128
    ori_subset_train_dataset = ori_train_datasets.select(range(nsamples))
    # print(type(ori_train_datasets))=<class 'datasets.arrow_dataset.Dataset'>
    # dataset = datasets.load_dataset(ori_train_datasets)  # 假设您已经加载了数据集

    # 获取前128个样本
    # ori_subset_train_dataset = dataset[:nsamples]
    # ori_subset_train_dataset = ori_train_datasets[:nsamples]
    print(len(ori_subset_train_dataset))

    # print(raw_datasets["train"][0])
    # examples = {'text': raw_datasets["train"]}
    # print(len(group_texts(examples,args.max_seq_length)))

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(
            utils.data.group_texts,
            fn_kwargs={
                'max_seq_length': max_seq_length,
            },
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    print(type(tokenized_datasets['train']))
    ori_subset_train_dataset_tokenized = tokenized_datasets['train'].select(range(nsamples))
    print(len(ori_subset_train_dataset_tokenized))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}. Decode to: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = utils.data.PTDataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                 mlm_probability=args.mlm_probability)

    print('train_dataset: ', len(train_dataset))
    # print(train_dataset[0])
    if args.max_train_samples is not None:
        # Number of samples might increase during Feature Creation, We select only specified max samples
        train_dataset = train_dataset.select(range(int(args.max_train_samples)))

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size,
        num_workers=0
    )

    train_dataloader_subset_dataset = train_dataset.select(range(int(10)))

    train_dataloader_subset = DataLoader(
        train_dataloader_subset_dataset, shuffle=True, collate_fn=data_collator, batch_size=1,
        num_workers=0
    )
    ori_subset_train_dataset_tokenized_das = DataLoader(
        ori_subset_train_dataset_tokenized, shuffle=True, collate_fn=data_collator, batch_size=1,
        num_workers=0
    )

    appr = Appr(args)
    acc=Acc(args)

    #计算当前数据在先前模型上的性能(ROC曲线)表现
    accuracy,auc_score=acc.acc_sim()
    # eval_roc=finetune.main(flag="eval_roc")
    print(f"accurary:{accuracy}")
    print(f"auc_score{auc_score}")

    appr.train(model, accelerator, train_dataset, train_dataloader, train_dataloader_subset,
               train_dataloader_subset_dataset, ori_subset_train_dataset, ori_subset_train_dataset_tokenized_das,auc_score)


if __name__ == "__main__":
    main()
