import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from networks import prompt

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        # assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last", "last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs,args):

        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states


        if args.contrast_type == 'add_hard_neg':
            n_sample = last_hidden.size(0)
            naive_last_hidden = last_hidden[:n_sample//3*2]
            hard_last_hidden = last_hidden[n_sample//3*2:]
            attention_mask = attention_mask[:n_sample//3*2]

            # print('naive_last_hidden: ',naive_last_hidden.size())
            # print('hard_last_hidden: ',hard_last_hidden.size())

            if self.pooler_type == "avg":
                naive_pool = ((naive_last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            if self.pooler_type in ['cls_before_pooler', 'cls']:
                naive_pool = naive_last_hidden[:, 0]
            hard_pool = hard_last_hidden[:,-args.n_tokens,:] # different pool results in different representation

            return torch.cat([naive_pool,hard_pool])



        elif args.contrast_type == 'naive':
            if self.pooler_type in ['cls_before_pooler', 'cls']:
                return last_hidden[:, 0]
            elif self.pooler_type == "avg":
                return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            elif self.pooler_type == "avg_first_last":
                first_hidden = hidden_states[0]
                last_hidden = hidden_states[-1]
                pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                return pooled_result
            elif self.pooler_type == "avg_top2":
                second_last_hidden = hidden_states[-2]
                last_hidden = hidden_states[-1]
                pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                return pooled_result
            elif self.pooler_type == "last":
                return last_hidden[:, -1]
            else:
                raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.args.pooler_type
    cls.pooler = Pooler(cls.args.pooler_type)
    if cls.args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.args.temp)
    cls.init_weights()


# def sequence_level_contrast(z2=None,z1=None,z3=None,sim=Similarity(temp=0.05)):


#     if z3 is not None:
#         z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
#         dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
#         z3_list[dist.get_rank()] = z3
#         z3 = torch.cat(z3_list, 0)

#     # Dummy vectors for allgather
#     z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
#     z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
#     # Allgather
#     dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
#     dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

#     # Since allgather results do not have gradients, we replace the
#     # current process's corresponding embeddings with original tensors
#     z1_list[dist.get_rank()] = z1
#     z2_list[dist.get_rank()] = z2
#     # Get full batch embeddings: (bs x N, hidden)
#     z1 = torch.cat(z1_list, 0)
#     z2 = torch.cat(z2_list, 0)

#     cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
#     # print('z1: ',z1.size())

#     # print('cos_sim: ',cos_sim.size())

#     # Hard negative

#     if z3 is not None:
#         z1_z3_cos = sim(z1.unsqueeze(1), z3.unsqueeze(0))
#         cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

#     # print('cos_sim: ',cos_sim.size())
#     labels = torch.arange(cos_sim.size(0)).long().cuda()
#     loss_fct = nn.CrossEntropyLoss()

#     if z3 is not None:
#         # Note that weights are actually logits of weights
#         z3_weight = 0
#         weights = torch.tensor(
#             [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
#         ).cuda()
#         cos_sim = cos_sim + weights

#     loss = loss_fct(cos_sim, labels)
#     return loss

def sequence_level_contrast(z2=None, z1=None, z3=None, sim=Similarity(temp=0.05)):
    # 假设 z1, z2, z3 已经是从同一个批次中提取的，且在同一个设备上

    # 计算 z1 和 z2 之间的余弦相似度
    # print(z1.device)
    # print(z2.device)
    cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))

    # 如果存在 z3，计算 z1 和 z3 之间的余弦相似度，并将其作为负样本添加到 cos_sim 中
    if z3 is not None:
        z1_z3_cos = sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], dim=1)

    # 创建一个与批次大小相同的标签向量，用于交叉熵损失函数
    labels = torch.arange(cos_sim.size(0)).long().to(z1.device)

    # 初始化交叉熵损失函数
    loss_fct = nn.CrossEntropyLoss()

    # 如果存在 z3，调整 cos_sim 中的权重以强调 z3 的影响
    if z3 is not None:
        # 创建一个权重矩阵，用于调整 cos_sim 中的值
        z3_weight = 0
        weights = torch.tensor([[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - 1)], device=z1.device)
        cos_sim = cos_sim + weights  # 将权重矩阵添加到 cos_sim

    # 计算损失
    loss = loss_fct(cos_sim, labels)

    return loss




