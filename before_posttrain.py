import logging
import math

import numpy as np
import os
from os import path
import torch
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from utils import utils
from networks.baselines import ewc, hat, softmask, memory, demix


def prepare(self, model, ori_subset_train_dataset, train_loader_subset, train_loader_subset_dataset, accelerator,
            ori_subset_train_dataset_tokenized_das,flag):
    self_fisher = None
    mask_pre = None
    mask_back = None
    buffer = None
    # TODO
    q_impt = None
    k_impt = None
    v_impt = None
    head_impt = None
    intermediate_impt = None
    output_impt = None

    if 'ewc' in self.args.baseline:
        if os.path.exists(os.path.join(self.args.output_dir + '../', 'fisher')):
            print('Load fisher matrix **************')
            self_fisher = torch.load(os.path.join(self.args.output_dir + '../', 'fisher'))
            for k, v in self_fisher.items():
                self_fisher[k] = self_fisher[k].cuda()

    elif 'adapter_hat' in self.args.baseline \
            or 'transformer_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:  # BCL included HAT
        if os.path.exists(os.path.join(self.args.output_dir + '../', 'mask_pre')):
            print('Load mask matrix **************')
            mask_pre = torch.load(os.path.join(self.args.output_dir + '../', 'mask_pre'))
            mask_back = torch.load(os.path.join(self.args.output_dir + '../', 'mask_back'))

            for k, v in mask_pre.items():
                mask_pre[k] = mask_pre[k].cuda()

            for k, v in mask_back.items():
                mask_back[k] = mask_back[k].cuda()

    elif 'derpp' in self.args.baseline:
        buffer = memory.Buffer(int(self.args.replay_sample_per_task * self.args.ntasks), args=self.args)
        if self.args.pt_task > 0:
            buffer.load(os.path.join(self.args.output_dir + '../', 'buffer'))

    elif self.args.pt_task > 0 and 'adapter_demix' in self.args.baseline:  # Initialize the new adapter using the nearest adapter
        model = demix.compute(train_loader_subset, train_loader_subset_dataset, model, accelerator, self.args)

    if 'dga' in self.args.baseline or 'das' in self.args.baseline:
        # train_loader_prune = accelerator.prepare(train_loader_subset)
        train_loader_prune = accelerator.prepare(ori_subset_train_dataset_tokenized_das)
        config = accelerator.unwrap_model(model).model.config
        if flag is None:
            if 'before_distill' in self.args.softmask_compute and (
                    self.args.pt_task == 0 or 'dga' in self.args.baseline):  # One and dga are the same
                #self.args.softmask_compute:before_distill_after_mlm
                config = accelerator.unwrap_model(model).model.config
                # softmask.compute_impt(args=self.args, config=config, model=model,
                #                                      eval_dataloader=train_loader_prune, accelerator=accelerator,
                #                                      prune_loss='before_distill')
                # print(f"模型：{model}")
                softmask.compute_impt_our_qkv(args=self.args, config=config, model=model,
                                              eval_dataloader=train_loader_prune, accelerator=accelerator,
                                              prune_loss='before_distill')

            # if 'before_mlm' in self.args.softmask_compute and self.args.pt_task == 0:  # only for wiki in task 0
            if 'before_mlm' in self.args.softmask_compute:  # only for wiki in task 0

                model = accelerator.prepare(model)
                # softmask.compute_impt(args=self.args, config=config, model=model,
                #                                      eval_dataloader=train_loader_prune, accelerator=accelerator,
                #                                      prune_loss='before_mlm')
                # print("进入了这里")
                softmask.compute_impt_our_qkv(args=self.args, config=config, model=model,
                                              eval_dataloader=train_loader_prune, accelerator=accelerator,
                                              prune_loss='before_mlm')

            accelerator.wait_for_everyone()
            # 重要性的计算
            q_impt_accumlate, k_impt_accumlate = softmask.accumulate_impt(self.args, config)
            return self, model, q_impt_accumlate, k_impt_accumlate, self_fisher, mask_pre, mask_back, buffer

        else:
            if 'curr_mlm' in self.args.softmask_compute:
                softmask.compute_impt_our_qkv_curr(args=self.args, config=config, model=model,
                                                                            eval_dataloader=train_loader_prune, accelerator=accelerator,
                                                                            prune_loss='curr_mlm')
            accelerator.wait_for_everyone()

            q_impt_list = []
            k_impt_list = []

            # for impt_dir_id, impt_dir in enumerate(self.args.saved_curr_output_dir):
            # print(f'Read importance from {self.args.saved_curr_output_dir}')

            q_impt_path_curr = f'{self.args.saved_curr_output_dir}/query_impt.npy'
            k_impt_path_curr = f'{self.args.saved_curr_output_dir}/key_impt.npy'

            if not path.exists(q_impt_path_curr):
                print(f'Warning: file {q_impt_path_curr} does not exist')

            q_impt_before = torch.Tensor(np.load(q_impt_path_curr)).cuda()
            # TODO:是否需要在这里进行初始化
            q_impt_before = softmask.impt_norm(q_impt_before)
            q_impt_list.append(q_impt_before)

            k_impt_before = torch.Tensor(np.load(k_impt_path_curr)).cuda()
            k_impt_before = softmask.impt_norm(k_impt_before)
            k_impt_list.append(k_impt_before)

            os.remove(q_impt_path_curr)
            os.remove(k_impt_path_curr)

            if len(q_impt_list) > 0:
                # q_impts.shape:[1,12,768,768]
                q_impts = torch.stack(q_impt_list)
                curr_q_impt, _ = q_impts.max(0)

                k_impts = torch.stack(k_impt_list)
                curr_k_impt, _ = k_impts.max(0)

            else:
                curr_q_impt, curr_k_impt = None, None

            return self, model, curr_q_impt, curr_k_impt, self_fisher, mask_pre, mask_back, buffer

        # accelerator.wait_for_everyone()

    #     #重要性的计算
    #     head_impt_accumlate, intermediate_impt_accumlate, output_impt_accumlate = softmask.accumulate_impt(self.args)
    #
    #     if accelerator.is_main_process:
    #         print(head_impt_accumlate.shape)
    #         print(f'Accumulated head layer importance: {head_impt_accumlate}')
    #         print(intermediate_impt_accumlate.shape)
    #         print(f'Accumulated intermediate layer importance: {intermediate_impt_accumlate}')
    #         print(output_impt_accumlate.shape)
    #         print(f'Accumulated output layer importance: {output_impt_accumlate}')
    #
    #     if 'head_mask' in self.args.layer_to_mask:
    #         head_impt = head_impt_accumlate
    #     if 'intermediate_mask' in self.args.layer_to_mask:
    #         intermediate_impt = intermediate_impt_accumlate
    #     if 'output_mask' in self.args.layer_to_mask:
    #         output_impt = output_impt_accumlate
    #
    # return self,model,head_impt, intermediate_impt, output_impt,self_fisher,mask_pre,mask_back,buffer

        # # 重要性的计算
        # q_impt_accumlate, k_impt_accumlate = softmask.accumulate_impt(self.args,config)

        # if accelerator.is_main_process:
        #     print(k_impt_accumlate.shape)
        #     print(f'Accumulated query layer importance: {q_impt_accumlate}')
        #     print(k_impt_accumlate.shape)
        #     print(f'Accumulated key layer importance: {k_impt_accumlate}')

            # print(v_impt_accumlate.shape)
            # print(f'Accumulated value layer importance: {v_impt_accumlate}')
            # print(intermediate_impt_accumlate.shape)
            # print(f'Accumulated intermediate layer importance: {intermediate_impt_accumlate}')
            # print(output_impt_accumlate.shape)
            # print(f'Accumulated output layer importance: {output_impt_accumlate}')

        # if 'q_mask' in self.args.layer_to_mask:
        #     q_impt = q_impt_accumlate
        # if 'k_mask' in self.args.layer_to_mask:
        #     k_impt = k_impt_accumlate
        # if 'v_mask' in self.args.layer_to_mask:
        #     v_impt = v_impt_accumlate
        # q_impt = q_impt_accumlate
        # k_impt = k_impt_accumlate
        # v_impt = v_impt_accumlate
        # if 'intermediate_mask' in self.args.layer_to_mask:
        #     intermediate_impt = intermediate_impt_accumlate
        # if 'output_mask' in self.args.layer_to_mask:
        #     output_impt = output_impt_accumlate

    # print("000")
    # print(f"type(q_impt):{type(q_impt)}")

    # return self, model, q_impt, k_impt, self_fisher, mask_pre, mask_back, buffer
