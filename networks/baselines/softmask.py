from tqdm.auto import tqdm
import torch
import os
import math
from torch.utils.data import DataLoader
import numpy as np
from networks.baselines.data import data_load
from .layerwrapper import WrappedGPT
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
from torch import nn
from itertools import zip_longest
import utils
import os
import torch.distributed as dist
import torch.autograd as autograd
from os import path


def gather_by_mean(head_impt):
    # head_impt_list = [torch.zeros_like(head_impt) for _ in range(dist.get_world_size())]
    # head_impt_list = [torch.zeros_like(head_impt) for _ in range(1)]
    # dist.all_gather(tensor_list=head_impt_list,
    #                 tensor=head_impt.contiguous())
    # dist.all_gather(tensor_list=head_impt_list,
    #                 tensor=head_impt)
    # head_impt_list = torch.stack(head_impt_list)
    # head_impt = torch.mean(head_impt_list, dim=0)
    head_impt_list = [head_impt]
    head_impt_list = torch.stack(head_impt_list)
    head_impt = torch.mean(head_impt_list, dim=0)

    return head_impt


def impt_norm(impt):
    tanh = torch.nn.Tanh()
    # normalized_impt = impt.clone()
    for layer in range(impt.size(0)):
        # mean=normalized_impt[layer].mean()
        # std=normalized_impt[layer].std(unbiased=False)+1e-7
        # normalized_impt[layer]=(normalized_impt[layer]-mean)/std
        impt[layer] = (impt[layer] - impt[layer].mean()) / impt[layer].std()+1e-7
        # impt[layer] = impt_normalized.clone()
        # impt[layer] = (impt[layer] - impt[layer].mean()) / std  # 2D, we need to deal with this for each layer
    impt = tanh(impt).abs()
    return impt
    # normalized_impt = tanh(normalized_impt).abs()
    # return normalized_impt

# def impt_norm(impt):
#     tanh = torch.nn.Tanh()
#     impt_mean = impt.mean(dim=1, keepdim=True)  # 沿着第2维计算每个样本的平均值
#     impt_std = impt.std(dim=1, keepdim=True)  # 沿着第2维计算每个样本的标准差
#     impt_normalized = (impt - impt_mean) / (impt_std + 1e-7)  # 防止除以0
#     # for layer in range(impt.size(0)):
#     #     impt[layer] = (impt[layer] - impt[layer].mean()) / impt[layer].std()  # 2D, we need to deal with this for each layer
#     impt = tanh(impt_normalized).abs()
#
#     return impt


def initial_impt(config):
    # 重要性矩阵初始化为 0，掩码矩阵初始化为 1
    n_encoder_layer, n_encoder_heads = config.num_hidden_layers, config.num_attention_heads

    intermediate_impt = torch.zeros(n_encoder_layer, config.intermediate_size).cuda()
    intermediate_mask = torch.ones(n_encoder_layer, config.intermediate_size).cuda()
    intermediate_mask.requires_grad_(requires_grad=True)

    output_impt = torch.zeros(n_encoder_layer, config.hidden_size).cuda()
    output_mask = torch.ones(n_encoder_layer, config.hidden_size).cuda()
    output_mask.requires_grad_(requires_grad=True)

    head_impt = torch.zeros(n_encoder_layer, n_encoder_heads).cuda()
    head_mask = torch.ones(n_encoder_layer, n_encoder_heads).cuda()
    head_mask.requires_grad_(requires_grad=True)

    tot_tokens = 0.0

    return head_impt, intermediate_impt, output_impt, head_mask, intermediate_mask, output_mask, tot_tokens


def initial_impt_qkv(config, args):
    # 重要性矩阵初始化为 0，掩码矩阵初始化为 1
    n_encoder_layer, n_encoder_heads = config.num_hidden_layers, config.num_attention_heads

    q_impt = torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
    q_mask = torch.ones(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
    q_mask.requires_grad_(requires_grad=True)

    k_impt = torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
    k_mask = torch.ones(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
    k_mask.requires_grad_(requires_grad=True)

    v_impt = torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
    v_mask = torch.ones(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
    v_mask.requires_grad_(requires_grad=True)

    intermediate_impt = torch.zeros(n_encoder_layer, config.intermediate_size).cuda()
    intermediate_mask = torch.ones(n_encoder_layer, config.intermediate_size).cuda()
    intermediate_mask.requires_grad_(requires_grad=True)

    output_impt = torch.zeros(n_encoder_layer, config.hidden_size).cuda()
    output_mask = torch.ones(n_encoder_layer, config.hidden_size).cuda()
    output_mask.requires_grad_(requires_grad=True)

    tot_tokens = 0.0

    return q_impt, k_impt, v_impt, intermediate_impt, output_impt, q_mask, k_mask, v_mask, intermediate_mask, output_mask, tot_tokens

def initial_impt_qkv_curr(config):
    # 重要性矩阵初始化为 0，掩码矩阵初始化为 1
    n_encoder_layer = config.num_hidden_layers
    q_mask = torch.ones(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
    k_mask = torch.ones(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
    return q_mask,k_mask

# def compute_impt_own(args,config, model,eval_dataloader,accelerator,prune_loss=None):
#     print("loading calibdation data")
#     dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
#     print("dataset loading complete")
#     with torch.no_grad():
#         inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prepare_calibration_input(model, dataloader, config, args, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.roberta.encoder.layer

    dtype = next(iter(model.parameters())).dtype
    # "llama-7B"  model.seqlen=2048,model.config.hidden_size=4096  此处的2记得修改
    # max_seq_length=164
    inps = torch.zeros((nsamples, args.max_seq_length, config.hidden_size), dtype=dtype, device=args.device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    # 得到的初始状态是第一层的参数？？？
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])  # ! inps[0]存的layers[0]的参数

    print("dataloader数据长度" + str(len(dataloader)))
    print(type(dataloader[0]))
    print(len(dataloader[0]))
    # print(dataloader)
    # batch为list，里面每个为tuple
    # for data in dataloader:
    # type(data[0]）=class torch.Tensor
    # print(type(data[0]))
    # data[0].shape=[1,164]
    # print(data[0].shape)
    # len((data)=2
    # print(len((data)))
    # try:
    #     model(data[0].to(args.device))
    # except ValueError:
    #     pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def compute_impt_our_qkv(args, config, model, eval_dataloader, accelerator, prune_loss=None):
    # model.train() # Train mode results in NAN

    # # TODO:
    # # MLM/Distill loss *****************************
    # q_impt, k_impt, v_impt, intermediate_impt, output_impt, \
    # q_mask, k_mask, v_mask, intermediate_mask, output_mask, tot_tokens = initial_impt_qkv(config, args)
    #
    # # 根据梯度回传计算重要性
    # print("q_mask")
    # # [12,768,768]
    # print(q_mask.shape)
    # print("k_mask")
    # # [12,768,768]
    # print(k_mask.shape)
    #
    # # TODO : 用DAS的dataloader过一遍模型,拿出每一次hidden_states，然后打印看一眼
    # for step, inputs in enumerate(tqdm(eval_dataloader, desc=f"Iteration {prune_loss}")):
    #     outputs = model(inputs, output_hidden_states=True, output_attentions=True,
    #                     q_mask=q_mask, k_mask=k_mask,
    #                     output_mask=output_mask,
    #                     intermediate_mask=intermediate_mask,
    #                     # v_mask=v_mask,intermediate_mask=intermediate_mask,output_mask=output_mask,
    #                     prune_loss=prune_loss)
    #     # TODO
    #     loss = outputs.loss
    #     # print(f"loss:{loss}")
    #     accelerator.backward(loss)
    #     # print(f"q_mask.shape:{q_mask.shape}")
    #     # q_impt += q_mask.grad.detach()
    #     # k_impt += k_mask.grad.detach()
    #     intermediate_impt +=intermediate_mask.grad.detach()
    #     output_impt+=output_mask.grad.detach()
    #     tot_tokens += inputs["attention_mask"].float().detach().sum().data
    #
    # q_impt /= tot_tokens
    # k_impt /= tot_tokens
    # intermediate_impt /= tot_tokens
    # output_impt /= tot_tokens
    # accelerator.wait_for_everyone()
    #
    # print(f"q_impt.shape:{q_impt.shape}")
    # print(f"q_impt:{q_impt}")
    # print(f"k_impt.shape:{k_impt.shape}")
    # print(f"k_impt:{k_impt}")
    #
    # if accelerator.is_main_process:
    #     np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/query_impt.npy', q_impt.detach().cpu().numpy())
    #     np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/key_impt.npy', k_impt.detach().cpu().numpy())
    #     np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/intermediate_impt.npy',intermediate_impt.detach().cpu().numpy())
    #     np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/output_impt.npy', output_impt.detach().cpu().numpy())
    #
    # return q_impt, k_impt,intermediate_impt,output_impt

    # MLM/Distill loss *****************************
    q_impt, k_impt, v_impt, intermediate_impt, output_impt,\
    q_mask, k_mask, v_mask, intermediate_mask, output_mask,tot_tokens = initial_impt_qkv(config,args)

    #print(f"args:{args}")

    #TODO
    # #根据梯度回传计算重要性
    # print("q_mask")
    # #[12,768,768]
    # print(q_mask.shape)
    # print("k_mask")
    # #[12,768,768]
    # print(k_mask.shape)

    # print("v_mask")
    # #[12,768,768]
    # print(v_mask.shape)
    # #[12,3072]
    # print("intermediate_mask")
    # print(intermediate_mask.shape)
    # #[12,768]
    # print("output_mask")
    # print(output_mask.shape)

    # print("loading calibdation data")
    # seed=0
    # nsamples=len(ori_subset_train_dataset)
    # dataloader,_=data_load(ori_subset_train_dataset,nsamples,seed,args.max_seq_length,args.tokenizer)

    # tmp=1
    # nsamples=0
    # #数值13需要根据模型的层数进行修改
    # scaler_row_list=[torch.zeros([768]) for _ in range(13)]

    tmp=1
    nsamples=0
    q_tmp=1
    q_nsamples=0
    k_tmp=1
    k_nsamples=0
    scaler_row_list=[torch.zeros([768]) for _ in range(13)]
    q_list = [torch.zeros([768,768]) for _ in range(12)]
    k_list = [torch.zeros([768, 768]) for _ in range(12)]
    # TODO : 用DAS的dataloader过一遍模型,拿出每一次hidden_states，然后打印看一眼

    #TODO：缩进
    # inputs=eval_dataloader.data
    for step, inputs in enumerate(tqdm(eval_dataloader, desc=f"Iteration {prune_loss}")):
        outputs = model(inputs,output_hidden_states=True,output_attentions=True,
                        q_mask=q_mask,k_mask=k_mask,
                        # v_mask=v_mask,intermediate_mask=intermediate_mask,output_mask=output_mask,
                        prune_loss=prune_loss)
        # #TODO
        # loss = outputs.loss
        # accelerator.backward(loss)
        #
        # intermediate_impt += intermediate_mask.grad.detach()
        # output_impt += output_mask.grad.detach()
        tot_tokens += inputs["attention_mask"].float().detach().sum().data

        #长度为13，[13,1,164,768]
        h_s = outputs.hidden_states
        #长度为12，[1，12，164，164]
        # o_a=outputs.attentions
        ##[batchsize,164,768]
        #print(len(h_s))

        #TODO:h_s为每层q,k,v的输入
        tensor_list=[]
        for i in range(len(h_s)):
            # 按照第一维求平均
            mean_tensor = torch.mean(h_s[i], dim=0, keepdim=True)
            tensor_list.append(mean_tensor.view(164, 768))

        #按层进行处理
        for j in range(len(tensor_list)):
            columns = tensor_list[j].shape[1]
            scaler_row = torch.zeros((columns), device=args.device)

            #inp = tensor_list[j].t()
            inp = tensor_list[j]
            # print(f"inp.shape:{inp.shape}")
            # print(f"inp:{inp}")
            #TODO
            # scaler_row = scaler_row_list[j] * nsamples / (nsamples+tmp)
            # nsamples += tmp
            # scaler_row + = torch.norm(inp, p=2, dim=0)  / nsamples

            scaler_row = torch.norm(inp, p=2, dim=0)
            scaler_row_list[j]=scaler_row

            #print(f"scaler_row:{scaler_row}")

            #  #计算每列的平均值,列平均的方法
            # column_means = torch.mean(inp, dim=0)

            # # 计算 inp 与 mean 之间的欧几里得距离
            # distances = np.linalg.norm(inp.detach().to("cpu") - column_means.detach().to("cpu"), axis=1)
            # result=torch.from_numpy(distances)
            # scaler_row= result.to(args.device)  / nsamples
            # scaler_row_list[j]=scaler_row

    #TODO:缩进

        layers = model.model.roberta.encoder.layer # TODO : 打印下shape，理想情况应该是12个

        # q_list=[]
        # k_list=[]
        # v_list=[]
        for i in range(len(layers)):
            layer = layers[i]
            # TODO : 想算头的重要性，获取到h_s和q k v 的矩阵后，W_metric 挨个算
            # copy
            subset = find_layers(layer)

            for name in subset:
                # print(f"impt layer {i} name {name}")

                if name in("attention.self.query"):
                    # print(f"wij相关信息:{subset[name].weight.data}")
                    # print(f"input相关信息：{scaler_row_list[i]}")
                    # W_metric = torch.abs(subset[name].weight.data.to(args.device)) * torch.abs(scaler_row_list[i])
                    q_metric = (q_list[i] * q_nsamples / (q_nsamples+q_tmp)).to(args.device)
                    q_nsamples += q_tmp
                    W_metric = torch.abs(subset[name].weight.data.to(args.device))* scaler_row_list[i]/q_nsamples
                    q_metric+=W_metric.to(args.device)
                    q_list[i] = q_metric
                    #print(f"{name}的重要性矩阵：{W_metric.shape}")
                    # q_list.append(W_metric)

                if name in("attention.self.key"):
                    # print(f"wij相关信息:{subset[name].weight.data}")
                    # print(f"input相关信息：{scaler_row_list[i]}")
                    # W_metric = torch.abs(subset[name].weight.data.to(args.device)) * torch.abs(scaler_row_list[i])
                    k_metric = (k_list[i] * k_nsamples / (k_nsamples + k_tmp)).to(args.device)
                    k_nsamples += k_tmp
                    W_metric = torch.abs(subset[name].weight.data.to(args.device)) * scaler_row_list[i] / k_nsamples
                    k_metric += W_metric.to(args.device)
                    k_list[i] = k_metric
                    #print(f"{name}的重要性矩阵：{W_metric.shape}")
                    # k_list.append(W_metric)

                # if name in("attention.value"):
                #     print(f"wij相关信息:{subset[name].weight.data.shape}")
                #     print(f"wij相关信息:{subset[name].weight.data}")
                #     print(f"input相关信息：{scaler_row_list[i].shape}")
                #     print(f"input相关信息：{scaler_row_list[i]}")
                #     # W_metric = torch.abs(subset[name].weight.data.to(args.device)) * torch.abs(scaler_row_list[i])
                #     W_metric = subset[name].weight.data.to(args.device) * torch.sqrt(scaler_row_list[i])
                #     #print(f"{name}的重要性矩阵：{W_metric.shape}")
                #     #[12,768,64]
                #     v_list.append(W_metric)
        # TODO
        loss = outputs.loss
        accelerator.backward(loss)
    #根据q,k,v计算头的重要性
    q_impt = torch.stack(q_list)
    k_impt = torch.stack(k_list)
    # v_impt = torch.stack(v_list)

    # print(f"q_impt.shape:{q_impt.shape}")
    # print(f"q_impt:{q_impt}")
    # print(f"k_impt.shape:{k_impt.shape}")
    # print(f"k_impt:{k_impt}")

    # print(f"v_impt.shape:{v_impt.shape}")
    # print(f"v_impt:{v_impt}")

    # q_impt/=tot_tokens
    # k_impt/=tot_tokens

    # intermediate_impt /= tot_tokens
    # output_impt /= tot_tokens
    # accelerator.wait_for_everyone()

    # q_impt_thresholded = torch.where(q_impt > 0.5, torch.tensor(1,device=q_impt.device), torch.tensor(0,device=q_impt.device))
    # k_impt_thresholded = torch.where(k_impt > 0.5, torch.tensor(1,device=q_impt.device), torch.tensor(0,device=q_impt.device))

    if accelerator.is_main_process:
        # np.save(f'{args.impt_query_output_dir}/{prune_loss}{args.pt_task}_query_impt.npy',q_impt_thresholded.detach().cpu().numpy())
        # np.save(f'{args.impt_key_output_dir}/{prune_loss}{args.pt_task}_key_impt.npy', k_impt_thresholded.detach().cpu().numpy())
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/query_impt.npy', q_impt.detach().cpu().numpy())
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/key_impt.npy',k_impt.detach().cpu().numpy())
        # np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/value_impt.npy', v_impt.detach().cpu().numpy())
        #
        # np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/intermediate_impt.npy',intermediate_impt.detach().cpu().numpy())
        # np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/output_impt.npy', output_impt.detach().cpu().numpy())

    return q_impt, k_impt

def compute_impt_our_qkv_curr(args, config, model, eval_dataloader, accelerator, prune_loss=None):

    # MLM/Distill loss *****************************
    q_mask, k_mask = initial_impt_qkv_curr(config)

    q_tmp=1
    q_nsamples=0
    k_tmp=1
    k_nsamples=0
    scaler_row_list=[torch.zeros([768]) for _ in range(13)]
    q_list = [torch.zeros([768,768]) for _ in range(12)]
    k_list = [torch.zeros([768, 768]) for _ in range(12)]
    # TODO : 用DAS的dataloader过一遍模型,拿出每一次hidden_states，然后打印看一眼

    #TODO：缩进
    # inputs=eval_dataloader.data
    with torch.no_grad():
        inputs=eval_dataloader.data
        # for step, inputs in enumerate(tqdm(eval_dataloader, desc=f"Iteration {prune_loss}")):
        outputs = model(inputs,output_hidden_states=True,output_attentions=True,
                        q_mask=q_mask,k_mask=k_mask,
                        # v_mask=v_mask,intermediate_mask=intermediate_mask,output_mask=output_mask,
                        prune_loss=prune_loss)
        # #TODO
        # tot_tokens += inputs["attention_mask"].float().detach().sum().data

        #长度为13，[1,164,768]
        h_s = outputs.hidden_states

        #TODO:h_s为每层q,k,v的输入
        tensor_list=[]
        for i in range(len(h_s)):
            # 按照第一维求平均
            mean_tensor = torch.mean(h_s[i], dim=0, keepdim=True)
            tensor_list.append(mean_tensor.view(164, 768))

        #按层进行处理
        for j in range(len(tensor_list)):
            columns = tensor_list[j].shape[1]
            scaler_row = torch.zeros((columns), device=args.device)

            inp = tensor_list[j]

            scaler_row = torch.norm(inp, p=2, dim=0)
            scaler_row_list[j]=scaler_row


        #TODO:缩进

        layers = model.model.roberta.encoder.layer # TODO : 打印下shape，理想情况应该是12个

        for i in range(len(layers)):
            layer = layers[i]
            # TODO : 想算头的重要性，获取到h_s和q k v 的矩阵后，W_metric 挨个算
            # copy
            subset = find_layers(layer)

            for name in subset:

                if name in("attention.self.query"):

                    q_metric = (q_list[i] * q_nsamples / (q_nsamples+q_tmp)).to(args.device)
                    q_nsamples += q_tmp
                    W_metric = torch.abs(subset[name].weight.data.to(args.device))* scaler_row_list[i]/q_nsamples
                    q_metric+=W_metric.to(args.device)
                    q_list[i] = q_metric

                if name in("attention.self.key"):
                    k_metric = (k_list[i] * k_nsamples / (k_nsamples + k_tmp)).to(args.device)
                    k_nsamples += k_tmp
                    W_metric = torch.abs(subset[name].weight.data.to(args.device)) * scaler_row_list[i] / k_nsamples
                    k_metric += W_metric.to(args.device)
                    k_list[i] = k_metric

        # TODO
        # loss = outputs.loss
        # accelerator.backward(loss)

        #根据q,k,v计算头的重要性
        q_impt = torch.stack(q_list)
        k_impt = torch.stack(k_list)

        q_impt_curr = impt_norm(q_impt)
        k_impt_curr = impt_norm(k_impt)

        if accelerator.is_main_process:

            np.save(f'{args.saved_curr_output_dir}/query_impt.npy', q_impt_curr.detach().cpu().numpy())
            np.save(f'{args.saved_curr_output_dir}/key_impt.npy',k_impt_curr.detach().cpu().numpy())

        del h_s,q_mask,k_mask,scaler_row_list,q_list,k_list,q_impt,k_impt,tensor_list
        torch.cuda.empty_cache()

    return q_impt_curr, k_impt_curr

def compute_impt_our_qkv_after(args, config, model, eval_dataloader, accelerator, prune_loss=None):
    # MLM/Distill loss *****************************
    q_mask, k_mask = initial_impt_qkv_curr(config)

    q_tmp = 1
    q_nsamples = 0
    k_tmp = 1
    k_nsamples = 0
    scaler_row_list = [torch.zeros([768]) for _ in range(13)]
    q_list = [torch.zeros([768, 768]) for _ in range(12)]
    k_list = [torch.zeros([768, 768]) for _ in range(12)]
    # TODO : 用DAS的dataloader过一遍模型,拿出每一次hidden_states，然后打印看一眼

    # TODO：缩进
    # inputs=eval_dataloader.data
    with torch.no_grad():
        # inputs = eval_dataloader.data
        for step, inputs in enumerate(tqdm(eval_dataloader, desc=f"Iteration {prune_loss}")):
            outputs = model(inputs, output_hidden_states=True, output_attentions=True,
                            q_mask=q_mask, k_mask=k_mask,
                            # v_mask=v_mask,intermediate_mask=intermediate_mask,output_mask=output_mask,
                            prune_loss=prune_loss)
            # #TODO
            # tot_tokens += inputs["attention_mask"].float().detach().sum().data

            # 长度为13，[1,164,768]
            h_s = outputs.hidden_states

            # TODO:h_s为每层q,k,v的输入
            tensor_list = []
            for i in range(len(h_s)):
                # 按照第一维求平均
                mean_tensor = torch.mean(h_s[i], dim=0, keepdim=True)
                tensor_list.append(mean_tensor.view(164, 768))

            # 按层进行处理
            for j in range(len(tensor_list)):
                columns = tensor_list[j].shape[1]
                scaler_row = torch.zeros((columns), device=args.device)

                inp = tensor_list[j]

                scaler_row = torch.norm(inp, p=2, dim=0)
                scaler_row_list[j] = scaler_row

            # TODO:缩进

            layers = model.model.roberta.encoder.layer  # TODO : 打印下shape，理想情况应该是12个

            for i in range(len(layers)):
                layer = layers[i]
                # TODO : 想算头的重要性，获取到h_s和q k v 的矩阵后，W_metric 挨个算
                # copy
                subset = find_layers(layer)

                for name in subset:

                    if name in ("attention.self.query"):
                        q_metric = (q_list[i] * q_nsamples / (q_nsamples + q_tmp)).to(args.device)
                        q_nsamples += q_tmp
                        W_metric = torch.abs(subset[name].weight.data.to(args.device)) * scaler_row_list[i] / q_nsamples
                        q_metric += W_metric.to(args.device)
                        q_list[i] = q_metric

                    if name in ("attention.self.key"):
                        k_metric = (k_list[i] * k_nsamples / (k_nsamples + k_tmp)).to(args.device)
                        k_nsamples += k_tmp
                        W_metric = torch.abs(subset[name].weight.data.to(args.device)) * scaler_row_list[i] / k_nsamples
                        k_metric += W_metric.to(args.device)
                        k_list[i] = k_metric

        # TODO
        # loss = outputs.loss
        # accelerator.backward(loss)

        # 根据q,k,v计算头的重要性
        q_impt = torch.stack(q_list)
        k_impt = torch.stack(k_list)

        q_impt_after = impt_norm(q_impt)
        k_impt_after= impt_norm(k_impt)

        if accelerator.is_main_process:
            np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/query_impt.npy', q_impt_after.detach().cpu().numpy())
            np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/key_impt.npy', k_impt_after.detach().cpu().numpy())

        del h_s, q_mask, k_mask, scaler_row_list, q_list, k_list, q_impt, k_impt, tensor_list
        torch.cuda.empty_cache()


    return q_impt_after, k_impt_after

def compute_impt_our(args, config, model, ori_subset_train_dataset_tokenized_das, accelerator, prune_loss=None):
    # model.train() # Train mode results in NAN

    # MLM/Distill loss *****************************
    head_impt, intermediate_impt, output_impt, \
    head_mask, intermediate_mask, output_mask, tot_tokens = initial_impt(config)

    # print(f"args:{args}")

    # 根据梯度回传计算重要性
    print("head_mask")
    # [12,12]
    print(head_mask.shape)
    print("intermediate_mask")
    # [12,3072]
    print(intermediate_mask.shape)
    print("output_mask")
    # [12,768]
    print(output_mask.shape)

    print("loading calibdation data")
    # seed=0
    # nsamples=len(ori_subset_train_dataset)
    # dataloader,_=data_load(ori_subset_train_dataset,nsamples,seed,args.max_seq_length,args.tokenizer)

    tmp = 1
    nsamples = 0
    # 数值13需要根据模型的层数进行修改
    scaler_row_list = [torch.zeros([768]) for _ in range(13)]
    # TODO : 用DAS的dataloader过一遍模型,拿出每一次hidden_states，然后打印看一眼
    for step, inputs in enumerate(tqdm(ori_subset_train_dataset_tokenized_das, desc=f"Iteration {prune_loss}")):
        outputs = model(inputs,
                        head_mask=head_mask, intermediate_mask=intermediate_mask, output_mask=output_mask,
                        output_hidden_states=True,
                        output_attentions=True, prune_loss=prune_loss)
        loss = outputs.loss

        # One can also only deal with the auto grad
        # g, = autograd.grad(loss, head_mask)
        # head_impt+= g.squeeze().detach()

        accelerator.backward(loss)

        intermediate_impt += intermediate_mask.grad.detach()
        output_impt += output_mask.grad.detach()

        tot_tokens += inputs["attention_mask"].float().detach().sum().data
        # print(outputs)
        # tuple 13
        # print(type(h_s))
        # print(len(h_s))
        # [1,164,768]

        h_s = outputs.hidden_states

        att = outputs.attentions
        # len=12
        # print(len(att))
        # #shape=[batchsize,num_head,seq,seq]  [1,12,164,164]
        # print(f"attention.shape:{att[0].shape}")

        tensor_list = []
        for i in range(len(h_s)):
            tensor_list.append(h_s[i].view(164, 768))

        for j in range(len(tensor_list)):
            columns = tensor_list[j].shape[1]
            scaler_row = torch.zeros((columns), device=args.device)

            inp = tensor_list[j].t()

            scaler_row = scaler_row_list[j] * nsamples / (nsamples + tmp)
            nsamples += tmp

            # 计算每列的平均值,列平均的方法
            column_means = torch.mean(inp, dim=0)

            # 计算 inp 与 mean 之间的欧几里得距离
            distances = np.linalg.norm(inp.detach().to("cpu") - column_means.detach().to("cpu"), axis=1)
            result = torch.from_numpy(distances)
            # print(scaler_row.device)
            # print(result.device)
            scaler_row = result.to(args.device) / nsamples
            scaler_row_list[j] = scaler_row
            # print(f"scaler_row_list[j]:{scaler_row_list[i]}")

    # print(len(scaler_row_list))
    # print(scaler_row_list[0].shape)

    print("dataset loading complete")

    # with torch.no_grad():
    #     inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, config, args,nsamples)
    # print(inps.shape)

    layers = model.model.roberta.encoder.layer  # TODO : 打印下shape，理想情况应该是12个

    intermediate_list = []
    output_list = []
    q_list = []
    k_list = []
    v_list = []
    for i in range(len(layers)):
        layer = layers[i]
        # TODO : 想算头的重要性，获取到h_s和q k v 的矩阵后，W_metric 挨个算
        # copy
        subset = find_layers(layer)

        for name in subset:
            print(f"impt layer {i} name {name}")

            print(f"wij相关信息:{subset[name].weight.data.shape}")

            # print(f"input相关信息:{scaler_row_list[i].reshape(1,-1).shape}")

            if name in ("attention.self.query"):
                W_metric = torch.abs(subset[name].weight.data.to(args.device)) * torch.abs(scaler_row_list[i])
                print(f"{name}的重要性矩阵：{W_metric.shape}")
                # [12,768,64]
                new_W_metric = W_metric.view(config.num_attention_heads, W_metric.shape[0], -1)

                # 对第二维和第三维的所有元素求和
                sum_metric = torch.sum(new_W_metric, dim=(1, 2))

                # 归一化求和结果
                # 首先计算所有元素的总和
                total_sum = torch.sum(sum_metric)

                # 然后对每个求和结果进行归一化
                normalized_q = sum_metric / total_sum
                # print(f"normalized_q:{normalized_q.shape}")
                q_list.append(normalized_q)

            if name in ("attention.self.key"):
                W_metric = torch.abs(subset[name].weight.data.to(args.device)) * torch.abs(scaler_row_list[i])
                print(f"{name}的重要性矩阵：{W_metric.shape}")
                # [12,768,64]
                new_W_metric = W_metric.view(config.num_attention_heads, W_metric.shape[0], -1)

                # 对第二维和第三维的所有元素求和
                sum_metric = torch.sum(new_W_metric, dim=(1, 2))

                # 归一化求和结果
                # 首先计算所有元素的总和
                total_sum = torch.sum(sum_metric)

                # 然后对每个求和结果进行归一化
                normalized_k = sum_metric / total_sum
                # print(f"normalized_k:{normalized_k}.shape")
                k_list.append(normalized_k)

            if name in ("attention.self.value"):
                W_metric = torch.abs(subset[name].weight.data.to(args.device)) * torch.abs(scaler_row_list[i])
                # print(f"{name}的重要性矩阵：{W_metric.shape}")
                # [12,768,64]
                new_W_metric = W_metric.view(config.num_attention_heads, W_metric.shape[0], -1)

                # 对第二维和第三维的所有元素求和
                sum_metric = torch.sum(new_W_metric, dim=(1, 2))

                # 归一化求和结果
                # 首先计算所有元素的总和
                total_sum = torch.sum(sum_metric)

                # 然后对每个求和结果进行归一化
                normalized_v = sum_metric / total_sum
                # print(f"normalized_v:{normalized_v.shape}")
                v_list.append(normalized_v)

            if name in ("intermediate.dense"):
                intermediate_list.append(subset[name].weight.data)

            if name in ("output.dense"):
                output_list.append(subset[name].weight.data)
    # 根据q,k,v计算头的重要性
    q_tensors = torch.stack(q_list)
    k_tensors = torch.stack(k_list)
    v_tensors = torch.stack(v_list)

    # 对应元素相加
    sum_tensors = q_tensors + k_tensors + v_tensors

    # 对求和结果进行平均
    head_impt = sum_tensors / 3

    print(f"head_impt.shape:{head_impt.shape}")
    print(f"head_impt:{head_impt}")

    # # Normalize
    # intermediate_impt /= tot_tokens
    # output_impt /= tot_tokens

    # print(f"intermediate_impt:{intermediate_impt}")
    # print(f"output_impt:{output_impt}")

    # accelerator.wait_for_everyone()

    # print(f"len(intermediate_list):{len(intermediate_list)}")
    # normalized_intermediate_list=[]
    # for i in range(len(intermediate_list)):
    #     # 按行求和
    #     sum_values = intermediate_list[i].sum(dim=1)
    #     #print(f"sum_values.shape:{sum_values.shape}")
    #
    #     # 计算张量的总和
    #     total = sum_values.sum()
    #     #print(f"total:{total}")
    #
    #     # 归一化张量的每个元素
    #     intermediate_normalized = sum_values / total
    #
    #     # # 归一化
    #     # intermediate_normalized = intermediate_list[i] / sum_values.unsqueeze(1)
    #     #print(f"intermediate_normalized.shape:{intermediate_normalized.shape}")
    #
    #     # 记录归一化的 intermediate
    #     normalized_intermediate_list.append(intermediate_normalized)
    # # 将归一化的 intermediate 列表转换为 [12, 3072] 形状
    # intermediate_impt = torch.stack(normalized_intermediate_list, 0)
    # # print(f"intermediate_impt.shape:{intermediate_impt.shape}")
    # # print(f"intermediate_impt:{intermediate_impt}")
    #
    # #print(f"len(output_list):{len(output_list)}")
    # normalized_output_list=[]
    # for i in range(len(output_list)):
    #     # 按行求和
    #     sum_values = output_list[i].sum(dim=1)
    #     #print(f"sum_values.shape:{sum_values.shape}")
    #
    #     # 计算张量的总和
    #     total = sum_values.sum()
    #     #print(f"total:{total}")
    #     # 归一化张量的每个元素
    #     output_normalized = sum_values / total
    #
    #     # # 归一化
    #     # intermediate_normalized = intermediate_list[i] / sum_values.unsqueeze(1)
    #     #print(f"output_normalized.shape:{output_normalized.shape}")
    #
    #     # 记录归一化的 intermediate
    #     normalized_output_list.append(output_normalized)
    # # 将归一化的 intermediate 列表转换为 [12, 3072] 形状
    # output_impt = torch.stack(normalized_output_list, 0)
    # # print(f"output_impt.shape:{output_impt.shape}")
    # # print(f"output_impt:{output_impt}")

    # Normalize
    intermediate_impt /= tot_tokens
    output_impt /= tot_tokens

    # print(f"intermediate_impt:{intermediate_impt}")
    # print(f"output_impt:{output_impt}")

    if accelerator.is_main_process:
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/head_impt.npy', head_impt.detach().cpu().numpy())
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/intermediate_impt.npy',
                intermediate_impt.detach().cpu().numpy())
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/output_impt.npy', output_impt.detach().cpu().numpy())

    return head_impt, intermediate_impt, output_impt


def compute_impt(args, config, model, eval_dataloader, accelerator, prune_loss=None):
    # model.train() # Train mode results in NAN

    # MLM/Distill loss *****************************
    head_impt, intermediate_impt, output_impt, \
    head_mask, intermediate_mask, output_mask, tot_tokens = initial_impt(config)

    # #根据梯度回传计算重要性
    # print("head_mask")
    # #[12,12]
    # print(head_mask.shape)
    # print("intermediate_mask")
    # #[12,3072]
    # print(intermediate_mask.shape)
    # print("output_mask")
    # #[12,768]
    # print(output_mask.shape)

    for step, inputs in enumerate(tqdm(eval_dataloader, desc=f"Iteration {prune_loss}")):
        outputs = model(inputs,
                        head_mask=head_mask, intermediate_mask=intermediate_mask, output_mask=output_mask,
                        prune_loss=prune_loss)
        loss = outputs.loss

        # [100,164]
        # print(inputs.input_ids.shape)

        # One can also only deal with the auto grad
        # g, = autograd.grad(loss, head_mask)
        # head_impt+= g.squeeze().detach()

        accelerator.backward(loss)

        head_impt += head_mask.grad.detach()
        intermediate_impt += intermediate_mask.grad.detach()
        output_impt += output_mask.grad.detach()

        tot_tokens += inputs["attention_mask"].float().detach().sum().data

    # Normalize
    head_impt /= tot_tokens

    intermediate_impt /= tot_tokens
    output_impt /= tot_tokens

    accelerator.wait_for_everyone()

    head_impt = gather_by_mean(head_impt)
    intermediate_impt = gather_by_mean(intermediate_impt)
    output_impt = gather_by_mean(output_impt)

    if accelerator.is_main_process:
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/head_impt.npy', head_impt.detach().cpu().numpy())
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/intermediate_impt.npy',
                intermediate_impt.detach().cpu().numpy())
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/output_impt.npy', output_impt.detach().cpu().numpy())

    return head_impt, intermediate_impt, output_impt


# def accumulate_impt(args):
#     head_impt_list = []
#     intermediate_impt_list = []
#     output_impt_list = []
#
#     for impt_dir_id, impt_dir in enumerate(args.saved_output_dir):
#         print(f'Read importance from {impt_dir}')
#
#         head_impt_path = f'{impt_dir}/head_impt.npy'
#         intermediate_impt_path = f'{impt_dir}/intermediate_impt.npy'
#         output_impt_path = f'{impt_dir}/output_impt.npy'
#
#         if not path.exists(head_impt_path):
#             print(f'Warning: file {head_impt_path} does not exist')
#             continue
#
#         head_impt = torch.Tensor(np.load(head_impt_path)).cuda()
#         head_impt = impt_norm(head_impt)
#         head_impt_list.append(head_impt)
#
#         intermediate_impt = torch.Tensor(np.load(intermediate_impt_path)).cuda()
#         intermediate_impt = impt_norm(intermediate_impt)
#         intermediate_impt_list.append(intermediate_impt)
#
#         output_impt = torch.Tensor(np.load(output_impt_path)).cuda()
#         output_impt = impt_norm(output_impt)
#         output_impt_list.append(output_impt)
#
#     if len(head_impt_list) > 0:
#         head_impts = torch.stack(head_impt_list)
#         head_impt, _ = head_impts.max(0)
#
#         intermediate_impts = torch.stack(intermediate_impt_list)
#         intermediate_impt, _ = intermediate_impts.max(0)
#
#         output_impts = torch.stack(output_impt_list)
#         output_impt, _ = output_impts.max(0)  # We take a max to accumulate
#
#     else:
#         head_impt, intermediate_impt, output_impt = None, None, None
#
#     return head_impt, intermediate_impt, output_impt


# def accumulate_impt(args):
#     q_impt_list = []
#     k_impt_list = []
#     v_impt_list = []
#     intermediate_impt_list = []
#     output_impt_list = []
#
#     for impt_dir_id, impt_dir in enumerate(args.saved_output_dir):
#         print(f'Read importance from {impt_dir}')
#
#         q_impt_path = f'{impt_dir}/query_impt.npy'
#         k_impt_path = f'{impt_dir}/key_impt.npy'
#         # v_impt_path = f'{impt_dir}/value_impt.npy'
#         # intermediate_impt_path = f'{impt_dir}/intermediate_impt.npy'
#         # output_impt_path = f'{impt_dir}/output_impt.npy'
#
#         if not path.exists(q_impt_path):
#             print(f'Warning: file {q_impt_path} does not exist')
#             continue
#
#         q_impt = torch.Tensor(np.load(q_impt_path)).cuda()
#         # q_impt.shape:[12,768,768]
#         q_impt = impt_norm(q_impt)
#         q_impt_list.append(q_impt)
#
#         k_impt = torch.Tensor(np.load(k_impt_path)).cuda()
#         k_impt = impt_norm(k_impt)
#         k_impt_list.append(k_impt)
#
#         #TODO:
#         if os.path.isfile(q_impt_path):
#             # 删除文件
#             os.remove(q_impt_path)
#
#         if os.path.isfile(k_impt_path):
#             # 删除文件
#             os.remove(k_impt_path)
#
#         # v_impt = torch.Tensor(np.load(v_impt_path)).cuda()
#         # v_impt = impt_norm(v_impt)
#         # v_impt_list.append(v_impt)
#         #
#         # intermediate_impt = torch.Tensor(np.load(intermediate_impt_path)).cuda()
#         # intermediate_impt = impt_norm(intermediate_impt)
#         # intermediate_impt_list.append(intermediate_impt)
#         #
#         # output_impt = torch.Tensor(np.load(output_impt_path)).cuda()
#         # output_impt = impt_norm(output_impt)
#         # output_impt_list.append(output_impt)
#
#     if len(q_impt_list) > 0:
#         # q_impts.shape:[1,12,768,768]
#         q_impts = torch.stack(q_impt_list)
#         q_impt, _ = q_impts.max(0)
#
#         k_impts = torch.stack(k_impt_list)
#         k_impt, _ = k_impts.max(0)
#
#         # v_impts = torch.stack(v_impt_list)
#         # v_impt, _ = v_impts.max(0)  # We take a max to accumulate
#         #
#         # intermediate_impts = torch.stack(intermediate_impt_list)
#         # intermediate_impt, _ = intermediate_impts.max(0)
#         #
#         # output_impts = torch.stack(output_impt_list)
#         # output_impt, _ = output_impts.max(0)
#
#     else:
#         q_impt, k_impt, v_impt = None, None, None
#
#     return q_impt, k_impt

# 定义一个函数来计算综合重要性权重
def calculate_combined_importance(A, B):
    # combined = torch.zeros_like(A)
    # combined[(A < 0.5) & (B < 0.5)] = 0
    # combined[(A < 0.5) & (B > 0.5)] = B[(A < 0.5) & (B > 0.5)]
    # combined[(A > 0.5) & (B < 0.5)] = 0
    # combined[(A > 0.5) & (B > 0.5)] = (A + B) / 2
    # 根据给定的规则计算综合重要性权重矩阵
    composite_weights = torch.where((A < 0.5) & (B < 0.5), torch.zeros_like(A),
                                    torch.where(A < 0.5, B,torch.where(B < 0.5, torch.zeros_like(A), (A + B) / 2)))
    return composite_weights

def accumulate_impt(args,config):
    q_impt_list = []
    k_impt_list = []
    # difference_q_impt_list=[]
    # difference_k_impt_list=[]

    #TODO
    # n_encoder_layer, n_encoder_heads = config.num_hidden_layers, config.num_attention_heads
    # q_impt_before=torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
    # k_impt_before=torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
    # q_impt_after = torch.zeros(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()
    # k_impt_after = torch.zeros(n_encoder_layer, config.hidden_size, config.hidden_size).cuda()

    for impt_dir_id, impt_dir in enumerate(args.saved_output_dir):
        print(f'Read importance from {impt_dir}')

        # if "before" in impt_dir:
        q_impt_path_before = f'{impt_dir}/query_impt.npy'
        k_impt_path_before = f'{impt_dir}/key_impt.npy'

        if not path.exists(q_impt_path_before):
            print(f'Warning: file {q_impt_path_before} does not exist')
            continue

        q_impt_before = torch.Tensor(np.load(q_impt_path_before)).cuda()
        #TODO:是否需要在这里进行初始化
        q_impt_before = impt_norm(q_impt_before)
        q_impt_list.append(q_impt_before)

        k_impt_before = torch.Tensor(np.load(k_impt_path_before)).cuda()
        k_impt_before = impt_norm(k_impt_before)
        k_impt_list.append(k_impt_before)

            # # TODO:
            # if os.path.isfile(q_impt_path_before):
            #     os.remove(q_impt_path_before)
            #
            # if os.path.isfile(k_impt_path_before):
            #     os.remove(k_impt_path_before)

        # if "after" in impt_dir:
        #     q_impt_path_after = f'{impt_dir}/query_impt.npy'
        #     k_impt_path_after = f'{impt_dir}/key_impt.npy'
        #
        #     if not path.exists(q_impt_path_after):
        #         print(f'Warning: file {q_impt_path_after} does not exist')
        #         continue
        #
        #     q_impt_after = torch.Tensor(np.load(q_impt_path_after)).cuda()
        #     q_impt_after = impt_norm(q_impt_after)
        #     # q_impt_list.append(q_impt_path_before)
        #
        #     k_impt_after = torch.Tensor(np.load(k_impt_path_after)).cuda()
        #     k_impt_after = impt_norm(k_impt_after)
            # k_impt_list.append(k_impt_before)

        #通过阈值权衡两个重要性矩阵，
        # 计算综合重要性权重
    # combined_q_impt = calculate_combined_importance(q_impt_after, q_impt_before)
    # difference_q_impt_list.append(combined_q_impt)
    # combined_k_impt = calculate_combined_importance(k_impt_after, k_impt_before)
    # difference_k_impt_list.append(combined_k_impt)

            # # 计算差值矩阵
            # difference_q_impt = q_impt_before - q_impt_after
            # # 根据条件调整差值矩阵,当差值为负且绝对差值大于0.3时，将差值矩阵中的值换成current原来的值,否则为差值的绝对值
            # adjusted_difference_q_impt = torch.where((difference_q_impt < 0) & (torch.abs(difference_q_impt) > 0.3), q_impt_before, torch.abs(difference_q_impt))
            # adjusted_difference_q_impt = impt_norm(adjusted_difference_q_impt)
            # difference_q_impt_list.append(adjusted_difference_q_impt)
            #
            # difference_k_impt = k_impt_before - k_impt_after
            # adjusted_difference_k_impt = torch.where((difference_k_impt < 0) & (torch.abs(difference_k_impt) > 0.3), k_impt_before, torch.abs(difference_k_impt))
            # adjusted_difference_k_impt = impt_norm(adjusted_difference_k_impt)
            # difference_k_impt_list.append(adjusted_difference_k_impt)

    # if len(difference_k_impt_list)>0:
    #     # q_impts.shape:[1,12,768,768]
    #     q_impts = torch.stack(difference_k_impt_list)
    #     q_impt, _ = q_impts.max(0)
    #
    #     k_impts = torch.stack(difference_k_impt_list)
    #     k_impt, _ = k_impts.max(0)

    if len(q_impt_list) > 0:
        # q_impts.shape:[1,12,768,768]
        q_impts = torch.stack(q_impt_list)
        q_impt, _ = q_impts.max(0)

        k_impts = torch.stack(k_impt_list)
        k_impt, _ = k_impts.max(0)

    else:
        q_impt, k_impt, v_impt = None, None, None

    return q_impt, k_impt

# def accumulate_impt_share(args,config,prune_loss=None):
#     q_impt_list = []
#     k_impt_list = []
#
#     #TODO
#     n_encoder_layer, n_encoder_heads = config.num_hidden_layers, config.num_attention_heads
#     q_impt_before=torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
#     k_impt_before=torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
#
#     share_k_impt=torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
#     uniq_k_impt=torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
#     share_q_impt=torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
#     uniq_q_impt=torch.zeros(n_encoder_layer, config.hidden_size,config.hidden_size).cuda()
#
#     query_target_file = None
#     # 存储每个tensor与目标tensor的交集
#     query_intersections = []
#     for filename in os.listdir(args.impt_query_output_dir):
#         if args.pt_task in filename:
#             # 拼接完整的文件路径
#             query_target_file = filename
#             break
#     query_target_data = torch.Tensor(np.load(os.path.join(args.impt_query_output_dir, query_target_file))).cuda()
#     q_impt=query_target_data
#
#     for filename in os.listdir(args.impt_query_output_dir):
#         if filename != query_target_file:
#             query_data = torch.Tensor(np.load(os.path.join(args.impt_query_output_dir, filename))).cuda()
#             # 计算交集
#             intersection = torch.logical_and(query_target_data, query_data)
#             query_intersections.append(intersection)
#
#     # 计算所有交集的并集
#     share_q_impt = query_intersections[0]
#     for query_intersection in query_intersections[1:]:
#         share_q_impt= torch.logical_or(share_q_impt, query_intersection)
#
#     uniq_q_impt=query_target_data - share_q_impt
#
#     #TODO
#     key_target_file = None
#     # 存储每个tensor与目标tensor的交集
#     key_intersections = []
#     for filename in os.listdir(args.impt_key_output_dir):
#         if args.pt_task in filename:
#             # 拼接完整的文件路径
#             key_target_file = filename
#             break
#     key_target_data = torch.Tensor(np.load(os.path.join(args.impt_key_output_dir, key_target_file))).cuda()
#     k_impt=key_target_data
#
#     for filename in os.listdir(args.impt_key_output_dir):
#         if filename != key_target_file:
#             key_data = torch.Tensor(np.load(os.path.join(args.impt_key_output_dir, filename))).cuda()
#             # 计算交集
#             intersection = torch.logical_and(key_target_data, key_data)
#             key_intersections.append(intersection)
#
#     # 计算所有交集的并集
#     share_k_impt = key_intersections[0]
#     for key_intersection in key_intersections[1:]:
#         share_k_impt = torch.logical_or(share_k_impt, key_intersection)
#
#     uniq_k_impt = key_target_data - share_k_impt
#
#     return share_q_impt, uniq_q_impt,share_k_impt,uniq_k_impt,q_impt,k_impt

# def soft_mask_gradient(model, pre_head_impt, pre_intermediate_impt,pre_output_impt,accelerator, epoch,step,args):
#
#     model_ori = accelerator.unwrap_model(model)
#
#     if accelerator.is_main_process and pre_head_impt is not None and epoch < 1 and step < 1:
#         if 'head_mask' in args.layer_to_mask:
#             print(f'Head mask usage {(pre_head_impt.sum() / pre_head_impt.numel()).item()}')
#         if 'intermediate_mask' in args.layer_to_mask:
#             print(f'Intermediate mask usage {(pre_intermediate_impt.sum() / pre_intermediate_impt.numel()).item()}')
#         if 'output_mask' in args.layer_to_mask:
#             print(f'Output mask usage {(pre_output_impt.sum() / pre_output_impt.numel()).item()}')
#
#     n_layers, n_heads = model_ori.model.config.num_hidden_layers, model_ori.model.config.num_attention_heads
#     head_size = int(model_ori.model.config.hidden_size / model_ori.model.config.num_attention_heads)
#
#     for layer in range(n_layers):
#
#         if 'head_mask' in args.layer_to_mask:
#             head_impt = pre_head_impt[layer].unsqueeze(-1).repeat((1, head_size))
#             head_impt = head_impt.flatten()
#
#             #梯度更新的地方
#             head_mask = 1 - head_impt
#
#             model_ori.model.roberta.encoder.layer[layer].attention.self.query.weight.grad *= head_mask
#             model_ori.model.roberta.encoder.layer[layer].attention.self.query.bias.grad *= head_mask
#
#             model_ori.model.roberta.encoder.layer[layer].attention.self.key.weight.grad *= head_mask
#             model_ori.model.roberta.encoder.layer[layer].attention.self.key.bias.grad *= head_mask
#
#             model_ori.model.roberta.encoder.layer[layer].attention.self.value.weight.grad *= head_mask
#             model_ori.model.roberta.encoder.layer[layer].attention.self.value.bias.grad *= head_mask
#
#             model_ori.model.roberta.encoder.layer[layer].attention.output.dense.weight.grad *= head_mask
#             model_ori.model.roberta.encoder.layer[layer].attention.output.dense.bias.grad *= head_mask
#
#         if 'intermediate_mask' in args.layer_to_mask:
#             intermediate_mask = (1 - pre_intermediate_impt[layer])
#             model_ori.model.roberta.encoder.layer[
#                 layer].intermediate.dense.weight.grad *= intermediate_mask.unsqueeze(1)
#             model_ori.model.roberta.encoder.layer[
#                 layer].intermediate.dense.bias.grad *= intermediate_mask
#
#         if 'output_mask' in args.layer_to_mask:
#             output_mask = (1 - pre_output_impt[layer])
#             model_ori.model.roberta.encoder.layer[
#                 layer].output.dense.weight.grad *= output_mask.unsqueeze(1)
#             model_ori.model.roberta.encoder.layer[layer].output.dense.bias.grad *= output_mask


# def soft_mask_gradient(model,pre_q_impt, pre_k_impt,accelerator, epoch, step, args):
def soft_mask_gradient(model,
                       # sigma_q_before, sigma_k_before, sigma_q_curr, sigma_k_curr,
                        q_impt, k_impt, accelerator, epoch,step, args):
    model_ori = accelerator.unwrap_model(model)

    #TODO:
    # if accelerator.is_main_process and pre_q_impt is not None and epoch < 1 and step < 1:
    #     if 'q_mask' in args.layer_to_mask:
    #         print(f'query mask usage {(pre_q_impt.sum() / pre_q_impt.numel()).item()}')
    #     if 'k_mask' in args.layer_to_mask:
    #         print(f'key mask usage {(pre_k_impt.sum() / pre_k_impt.numel()).item()}')

        # if 'v_mask' in args.layer_to_mask:
        #     print(f'value mask usage {(pre_v_impt.sum() / pre_v_impt.numel()).item()}')
        # if 'intermediate_mask' in args.layer_to_mask:
        #     print(f'Intermediate mask usage {(pre_intermediate_impt.sum() / pre_intermediate_impt.numel()).item()}')
        # if 'output_mask' in args.layer_to_mask:
        #     print(f'Output mask usage {(pre_output_impt.sum() / pre_output_impt.numel()).item()}')

    n_layers, n_heads = model_ori.model.config.num_hidden_layers, model_ori.model.config.num_attention_heads
    head_size = int(model_ori.model.config.hidden_size / model_ori.model.config.num_attention_heads)

    for layer in range(n_layers):

        if 'q_mask' in args.layer_to_mask:
            # q_mask [768,]
            # query.weight [768,768]
            # print(f"model_ori.model.roberta.encoder.layer[layer].attention.query.bias.grad:{model_ori.model.roberta.encoder.layer[layer].attention.query.bias.grad.shape}")
            # #TODO
            # # 创建布尔掩码，其中小于0.1的元素为True
            # mask = pre_q_impt[layer] < 0.1
            # # 将小于0.1的元素设置为0
            # pre_q_impt[layer][mask] = 0

            # a=model_ori.model.roberta.encoder.layer[layer].attention.self.sigma_q_before

            # expand_sigma_q_before=(model_ori.model.roberta.encoder.layer[
            #     layer].attention.self.sigma_q_before.pow(2)+1e-7)
            #     # .unsqueeze(0).expand(12, -1, -1)
            #
            # expand_sigma_q_curr=(model_ori.model.roberta.encoder.layer[
            #     layer].attention.self.sigma_q_curr.pow(2)+1e-7)
            #     # .unsqueeze(0).expand(12, -1, -1)
            # q_before=befor_q_impt[layer] / expand_sigma_q_before
            # q_curr=curr_q_impt[layer] / expand_sigma_q_curr
            # impt_q_combine=q_before+q_curr

            # impt_q_combine = (befor_q_impt[layer] /(model_ori.model.roberta.encoder.layer[
            #     layer].attention.self.sigma_q_before.pow(2)+1e-7))+(curr_q_impt[layer] /(model_ori.model.roberta.encoder.layer[
            #     layer].attention.self.sigma_q_curr.pow(2)+1e-7))
            # impt_q_combine=befor_q_impt[layer]/expand_sigma_q_before + curr_q_impt[layer]/expand_sigma_q_curr

            # q_mask = impt_norm(impt_q_combine)
            # q_mask = (q_impt_combine)
            # share_q_mask = (share_q_impt[layer])
            # uniq_q_mask=(uniq_q_impt[layer])
            # q_mask[q_mask < 0.5] = 0
            # q_mask[q_mask >= 0.5] = 1
            # del(impt_q_combine)


            # new code
            # 计算方式： (before+current/sigma^2)/2
            q_mask = q_impt[layer]
            model_ori.model.roberta.encoder.layer[
                layer].attention.self.query.weight.grad *= q_mask
            # model_ori.model.roberta.encoder.layer[
            #     layer].attention.self.query.bias.grad *= q_mask

        if 'k_mask' in args.layer_to_mask:
            # #TODO
            # # 创建布尔掩码，其中小于0.1的元素为True
            # mask = pre_k_impt[layer] < 0.1
            # # 将小于0.1的元素设置为0
            # pre_k_impt[layer][mask] = 0
            # share_k_mask = ( share_k_impt[layer])
            # uniq_k_mask=(uniq_k_impt[layer])
            # k_mask[k_mask < 0.5] = 0
            # k_mask[k_mask >= 0.5] = 1

            # expand_sigma_k_before = (model_ori.model.roberta.encoder.layer[
            # layer].attention.self.sigma_k_before.pow(2) + 1e-7)
            #     # .unsqueeze(0).expand(12, -1, -1)
            #
            # expand_sigma_k_curr = (model_ori.model.roberta.encoder.layer[
            # layer].attention.self.sigma_k_curr.pow(2) + 1e-7)
            #     # .unsqueeze(0).expand(12,-1, -1)
            # k_before = befor_k_impt[layer] / expand_sigma_k_before
            # k_curr = curr_k_impt[layer] / expand_sigma_k_curr
            # impt_k_combine = k_before + k_curr

            # impt_k_combine=(befor_k_impt[layer]/(model_ori.model.roberta.encoder.layer[
            # layer].attention.self.sigma_k_before.pow(2) + 1e-7))+(curr_k_impt[layer] /(model_ori.model.roberta.encoder.layer[
            # layer].attention.self.sigma_k_curr.pow(2) + 1e-7))

            # impt_k_combine = befor_k_impt[layer] / expand_sigma_k_before + curr_k_impt[layer] / expand_sigma_k_curr
            # impt_k_combine = befor_k_impt / model_ori.model.roberta.encoder.layer[
            #     layer].attention.self.sigma_k_before.weight.pow(2) + curr_k_impt / model_ori.model.roberta.encoder.layer[
            #     layer].attention.self.sigma_k_curr.weight.pow(2)
            # k_mask = impt_norm(impt_k_combine)
            # del(impt_k_combine)
            # k_mask=(k_impt_combine)

            # new code
            # 计算方式： (before+current/sigma^2)/2
            k_mask = k_impt[layer]

            model_ori.model.roberta.encoder.layer[
                layer].attention.self.key.weight.grad *= k_mask
            # model_ori.model.roberta.encoder.layer[
            #     layer].attention.self.key.bias.grad *= k_mask

        # if 'v_mask' in args.layer_to_mask:
        #     v_mask = (pre_v_impt[layer])
        #     model_ori.model.roberta.encoder.layer[
        #         layer].attention.value.weight.grad *= v_mask
        #     # model_ori.model.roberta.encoder.layer[
        #     #     layer].attention.value.bias.grad *= v_mask.view(-1)
        #
        # if 'intermediate_mask' in args.layer_to_mask:
        #     intermediate_mask = (1-pre_intermediate_impt[layer])
        #     model_ori.model.roberta.encoder.layer[
        #         layer].intermediate.dense.weight.grad *= intermediate_mask.unsqueeze(1)
        #     model_ori.model.roberta.encoder.layer[
        #         layer].intermediate.dense.bias.grad *= intermediate_mask
        #
        # if 'output_mask' in args.layer_to_mask:
        #     output_mask = (1-pre_output_impt[layer])
        #     model_ori.model.roberta.encoder.layer[
        #         layer].output.dense.weight.grad *= output_mask.unsqueeze(1)
        #     model_ori.model.roberta.encoder.layer[layer].output.dense.bias.grad *= output_mask

        # if 'q_mask' in args.layer_to_mask:
        #     q_impt = pre_q_impt[layer].unsqueeze(-1).repeat((1, head_size))
        #     q_impt = q_impt.flatten()

        #     #梯度更新的地方
        #     head_mask = 1 - q_impt

        #     model_ori.model.roberta.encoder.layer[layer].attention.self.query.weight.grad *= head_mask
        #     model_ori.model.roberta.encoder.layer[layer].attention.self.query.bias.grad *= head_mask

        #     model_ori.model.roberta.encoder.layer[layer].attention.self.key.weight.grad *= head_mask
        #     model_ori.model.roberta.encoder.layer[layer].attention.self.key.bias.grad *= head_mask

        #     model_ori.model.roberta.encoder.layer[layer].attention.self.value.weight.grad *= head_mask
        #     model_ori.model.roberta.encoder.layer[layer].attention.self.value.bias.grad *= head_mask

        #     model_ori.model.roberta.encoder.layer[layer].attention.output.dense.weight.grad *= head_mask
        #     model_ori.model.roberta.encoder.layer[layer].attention.output.dense.bias.grad *= head_mask