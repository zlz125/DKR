import copy
import shutil
import argparse
import logging
import math
import os
import random
import sys
import csv
import torch
import datasets
import transformers
from accelerate import Accelerator, DistributedType
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
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
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import utils.roberta
from approaches import after_posttrain, before_posttrain, compute_loss, compute_gradient, update_model
from networks.baselines import ewc, hat, softmask, memory

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class AutomaticWeightedGranditImp(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedGranditImp, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
    #
    #loss = f(loss1,loss2) = \frac{1}{a^{2}}loss1+log(1+a^2)+\frac{1}{b^{2}}loss2+1og(1+b^2)
    def forward(self, *x):
        result_list = []
        for i, importance in enumerate(x):
            result_list.append(1/(self.params[i] ** 2)*importance)
        return tuple(result_list)




class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.mask = utils.model.mask
        self.get_view_for = utils.model.get_view_for
        self.get_view_for_tsv = utils.model.get_view_for_tsv

        return

    def train(self, model, accelerator, train_dataset, train_loader, train_loader_subset, train_loader_subset_dataset,
              ori_subset_train_dataset, ori_subset_train_dataset_tokenized_das,auc_score):

        Inc_train_loss = []

        filename = f"{self.args.output_dir}/{self.args.dataset_name}_losses.csv"
        print(f"filename:{filename}")
        # with open(filename, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['Step', 'Loss'])


        #TODO
        awgi = AutomaticWeightedGranditImp(2)
        optimizer = utils.optimize.lookfor_optimize(model,self.args)

        # print(self.args)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_loader, train_loader_subset, ori_subset_train_dataset_tokenized_das = accelerator.prepare(
            model, optimizer, train_loader, train_loader_subset, ori_subset_train_dataset_tokenized_das)

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        if accelerator.distributed_type == DistributedType.TPU:
            model.tie_weights()

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)

        if self.args.max_samples is not None:
            self.args.max_train_steps = self.args.max_samples // (
                    self.args.per_device_train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps)

        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # Warm up can be important
        # warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
        self.args.num_warmup_steps = int(float(self.args.warmup_proportion) * float(self.args.max_train_steps))  # 0.1

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Train!
        total_batch_size = self.args.per_device_train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps

        # before training ***********************************************************************************************

        # self,model,head_impt, intermediate_impt, output_impt,self_fisher,mask_pre,mask_back,buffer \
        #     = before_posttrain.prepare(self,model,ori_subset_train_dataset,train_loader_subset,train_loader_subset_dataset, accelerator,ori_subset_train_dataset_tokenized_das)

        # #TODO
        config = accelerator.unwrap_model(model).model.config
        # n_encoder_layer, n_encoder_heads = config.num_hidden_layers, config.num_attention_heads
        # befor_q_impt = torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
        # befor_q_impt.requires_grad_(requires_grad=True)
        # befor_k_impt = torch.zeros(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
        # befor_k_impt.requires_grad_(requires_grad=True)
        # curr_q_impt = torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
        # curr_q_impt.requires_grad_(requires_grad=True)
        # curr_k_impt = torch.zeros(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
        # curr_k_impt.requires_grad_(requires_grad=True)


        # q_mask = torch.ones(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
        # q_mask.requires_grad_(requires_grad=True)
        # q_grad=torch.zeros(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
        #
        # k_mask = torch.ones(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
        # k_mask.requires_grad_(requires_grad=True)
        # k_grad = torch.zeros(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
        #
        # head_mask = torch.ones(n_encoder_layer, n_encoder_heads).cuda()
        # head_mask.requires_grad_(requires_grad=True)
        # head_grad=torch.zeros(n_encoder_layer, n_encoder_heads).cuda()

        # sigma_q_before = torch.ones(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
        # sigma_q_before.requires_grad_(requires_grad=True)
        # sigma_k_before = torch.ones(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
        # sigma_k_before.requires_grad_(requires_grad=True)
        # sigma_q_curr = torch.ones(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
        # sigma_q_curr.requires_grad_(requires_grad=True)
        # sigma_k_curr = torch.ones(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
        # sigma_k_curr.requires_grad_(requires_grad=True)

        # optimizer_other = utils.optimize.lookfor_optimize_other(external_params, self.args)

        # external_params, optimizer_other, train_loader, train_loader_subset, ori_subset_train_dataset_tokenized_das = accelerator.prepare(
        #     external_params, optimizer_other, train_loader, train_loader_subset, ori_subset_train_dataset_tokenized_das)

        # lr_scheduler_other = get_scheduler(
        #     name=self.args.lr_scheduler_type,
        #     optimizer=optimizer_other,
        #     num_warmup_steps=self.args.num_warmup_steps,
        #     num_training_steps=self.args.max_train_steps,
        # )
        flag=None

        self, model, befor_q_impt, befor_k_impt, self_fisher, mask_pre, mask_back, buffer \
            = before_posttrain.prepare(self, model, ori_subset_train_dataset, train_loader_subset,
                                       train_loader_subset_dataset, accelerator, ori_subset_train_dataset_tokenized_das,flag)
        # flag = "curr"
        # self, model, curr_q_impt, curr_k_impt, self_fisher, mask_pre, mask_back, buffer \
        #     = before_posttrain.prepare(self, model, ori_subset_train_dataset, train_loader_subset,
        #                                train_loader_subset_dataset, accelerator, ori_subset_train_dataset_tokenized_das, flag)
        # befor_q_impt=befor_q
        # befor_k_impt=befor_k

        # q_impt, k_impt, v_impt = torch.random(12,768,768)
        # print("111")
        # print(f"type(q_impt):{type(q_impt)}")
        # before training ***********************************************************************************************

        #TODO
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        # num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        # self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch

        if accelerator.is_main_process:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
            logger.info(f"  Total samples = {self.args.max_train_steps * total_batch_size}")
            logger.info(
                f"  Learning Rate = {self.args.learning_rate}, Warmup Num = {self.args.num_warmup_steps}, Pre-trained Model = {self.args.model_name_or_path}")
            logger.info(
                f"  Seq ID = {self.args.idrandom}, Task id = {self.args.pt_task}, dataset name = {self.args.dataset_name}")
            logger.info(f"  Baseline = {self.args.baseline}, Smax = {self.args.smax}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        global_step = 0  # This will be used by CLMOE if we choose 'auto_encoder' as the route type.

        writer = None
        if accelerator.is_main_process:
            tensorboard_file = os.path.join(self.args.output_dir, str(self.args.dataset_name) + '_log')
            print('tensorboard_file: ', tensorboard_file)
            if os.path.isdir(tensorboard_file):
                shutil.rmtree(tensorboard_file)
            writer = utils.model.setup_writer(tensorboard_file)

        try:
            if not self.args.eval_only:
                # TODO
                scaler = GradScaler()
                for epoch in range(self.args.num_train_epochs):
                    # break
                    model.train()
                    for step, batch in enumerate(train_loader):
                        # torch.autograd.set_detect_anomaly(True)
                        # self, model, outputs = compute_loss.compute(self,model,batch,head_impt,intermediate_impt,output_impt,self_fisher,mask_pre,train_loader,step,accelerator)
                        # self, model, outputs = compute_loss.compute(self, model, batch, head_impt, intermediate_impt,
                        #                                             output_impt, self_fisher, mask_pre, train_loader,
                        #                                             step, accelerator)

                        # TODO:
                        flag = "curr"
                        self, model, curr_q_impt, curr_k_impt,self_fisher, mask_pre, mask_back, buffer \
                            = before_posttrain.prepare(self, model, ori_subset_train_dataset, train_loader_subset,
                                                       train_loader_subset_dataset, accelerator, batch,flag)

                        # curr_q_impt,curr_k_impt,outputs=softmask.compute_impt_our_qkv_curr(args=self.args, config=config, model=model,
                        #                           eval_dataloader=batch, accelerator=accelerator,prune_loss='curr_mlm')

                        q_impt = softmask.impt_norm((befor_q_impt + curr_q_impt) / 2 * auc_score)
                        k_impt = softmask.impt_norm((befor_k_impt + curr_k_impt) / 2 * auc_score)

                        self, model, outputs = compute_loss.compute(self, model, batch,
                                                                    # sigma_q_before,sigma_k_before,sigma_q_curr,sigma_k_curr,
                                                                    q_impt, k_impt,
                                                                    # curr_q_impt, curr_k_impt,
                                                                    # q_impt,k_impt,
                                                                    # intermediate_impt, output_impt,
                                                                    self_fisher, mask_pre, train_loader, step,
                                                                    accelerator)
                        loss = outputs.loss

                        # flag="curr"
                        # self, model, curr_q_impt, curr_k_impt, outputs,self_fisher, mask_pre, mask_back, buffer \
                        #     = before_posttrain.prepare(self, model, ori_subset_train_dataset, train_loader_subset,
                        #                                train_loader_subset_dataset, accelerator, batch,flag)

                        # if 'curr_mlm' in self.args.softmask_compute:
                        #     config = accelerator.unwrap_model(model).model.config
                        #     curr_q_impt,curr_k_impt=softmask.compute_impt_our_qkv_curr(args=self.args, config=config, model=model,
                        #                           eval_dataloader=batch, accelerator=accelerator,
                        #                           prune_loss='curr_mlm')

                            # impt_q_combine=befor_q_impt/sigma_q_before.pow(2) + curr_q_impt/sigma_q_curr.pow(2)
                            # impt_k_combine = before_k_impt / sigma_k_before.pow(2) + curr_k_impt / sigma_k_curr.pow(2)

                            # q_impt_combine = softmask.impt_norm(impt_q_combine)
                            # k_impt_combine = softmask.impt_norm(impt_k_combine)

                        # curr_impt_tumple = awgi(curr_q_impt, curr_k_impt)
                        # curr_q = curr_impt_tumple[0]
                        # curr_k = curr_impt_tumple[1]

                        # loss=outputs.loss

                        # model = compute_gradient.compute(self,model,head_impt, intermediate_impt, output_impt,batch, loss,buffer,mask_back,outputs,epoch,step,accelerator)

                        model = compute_gradient.compute(self, model,
                                                         # sigma_q_before, sigma_k_before, sigma_q_curr, sigma_k_curr,
                                                         q_impt, k_impt,
                                                         # curr_q_impt, curr_k_impt,
                                                         # q_mask,k_mask,q_grad,k_grad,head_mask,head_grad,
                                                         # intermediate_impt, output_impt,
                                                         batch, loss, buffer, mask_back, outputs, epoch, step,
                                                         accelerator)

                        global_step += 1

                        Inc_train_loss.append(loss.item())

                        if step % self.args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                            update_model.update(self, model, optimizer,outputs, loss, writer, lr_scheduler,
                                                progress_bar, global_step, completed_steps, accelerator)
                            completed_steps += 1

                        # break
                        if completed_steps >= self.args.max_train_steps:
                            break

        except KeyboardInterrupt:  # even if contro-C, I still want to save model
            return

        # after_posttrain.compute(self, model, train_loader_subset, self_fisher,mask_pre, buffer,accelerator)
        after_posttrain.compute(self, model, ori_subset_train_dataset_tokenized_das, self_fisher, mask_pre, buffer,
                                accelerator)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Loss'])
            for i, loss in enumerate(Inc_train_loss):
                writer.writerow([i, loss])

