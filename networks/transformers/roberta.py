# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model."""

import torch
from torch.nn import functional as F
import math
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from typing import Callable
from networks.baselines import softmask

from transformers.modeling_outputs import ModelOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaSelfAttention, RobertaPooler, \
    RobertaPreTrainedModel, RobertaClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, \
    RobertaOutput, RobertaLayer, RobertaAttention, RobertaForMaskedLM, RobertaForSequenceClassification, \
    RobertaForTokenClassification

from networks.adapters.mixins.bert import (
    BertModelAdaptersMixin,
    BertOutputAdaptersMixin,
    BertSelfOutputAdaptersMixin,
)
import inspect

from networks.adapters.model_mixin import InvertibleAdaptersMixin, ModelWithHeadsAdaptersMixin
from networks.adapters.prefix_tuning import PrefixTuningShim
from transformers.adapters.composition import adjust_tensors_for_parallel
from transformers.adapters.context import ForwardContext
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from typing import Optional
from transformers.activations import ACT2FN

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]
import utils


# because adapter has been changed, we need to re-write the complete class


# Copied from transformers.models.modeling_bert.BertSelfOutput
class MyRobertaSelfOutput(BertSelfOutputAdaptersMixin, nn.Module):
    def __init__(self, config, args):
        BertSelfOutputAdaptersMixin.__init__(self, args)
        nn.Module.__init__(self)

        self.config = config

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_adapter_modules()

    def forward(self, hidden_states, input_tensor, down_mask, up_mask, **kwargs):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter_layer_forward(hidden_states, input_tensor, 'encoder', down_mask, up_mask,
                                                   self.LayerNorm)
        return hidden_states

# TODO:
# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Roberta
class OurRobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, location_key: Optional[str] = None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # self.output = MyRobertaSelfOutput(config, args)
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        #TODO
        # 定义sigma参数
        # self.curr_q_impt = nn.Parameter(
        #     torch.eye(self.all_head_size))
        # self.curr_q_impt.requires_grad = True
        #
        # self.curr_k_impt = nn.Parameter(
        #     torch.eye(self.all_head_size))
        # self.curr_k_impt.requires_grad = True
        # self.sigma = nn.Parameter(torch.ones(1))

        # self.sigma_q_curr = nn.Parameter(
        #     torch.ones(1))
        # self.sigma_q_curr.requires_grad = True
        #
        # self.sigma_k_curr = nn.Parameter(
        #     torch.ones(1))
        # self.sigma_k_curr.requires_grad = True

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

        self.prefix_tuning = PrefixTuningShim(location_key + "_prefix" if location_key else None, config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            q_mask=None,
            k_mask=None,
            v_mask=None,
            sigma_q_before=None,
            sigma_k_before=None,
            sigma_q_curr=None,
            sigma_k_curr=None,
            befor_q_mask=None,
            befor_k_mask=None,
            curr_q_mask=None,
            curr_k_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            down_mask=None,
            up_mask=None,
            **kwargs
            # TODO: +q_kmask 第二种 q_mask k_mask单独传
    ):
        if q_mask is not None:
            q_masked_weights = self.query.weight.data * q_mask
            self.query.weight = torch.nn.Parameter(q_masked_weights, requires_grad=self.query.weight.requires_grad)
        if k_mask is not None:
            k_masked_weights = self.key.weight.data * k_mask
            self.key.weight = torch.nn.Parameter(k_masked_weights, requires_grad=self.key.weight.requires_grad)

        #TODO
        mixed_query_layer = self.query(hidden_states)  #[1,164,768]

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            # TODO:走的这里
            key_layer = self.transpose_for_scores(self.key(hidden_states))  # [1,12,164,64]
            value_layer = self.transpose_for_scores(self.value(hidden_states))  # [1,12,164,64]

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # # TODO:此处更改key_layer,value_layer
        # if q_mask is not None:
        #     query_layer_permuted = query_layer.permute(0, 2, 1, 3)
        #     query_layer_reshaped = query_layer_permuted.view(query_layer_permuted.size(0), query_layer_permuted.size(1), -1)
        #     query_layer = query_layer_reshaped * q_mask
        #     query_layer = self.transpose_for_scores(query_layer)
        #
        # if k_mask is not None:
        #     key_layer_permuted = key_layer.permute(0, 2, 1, 3)
        #     key_layer_reshaped = key_layer_permuted.view(key_layer_permuted.size(0), key_layer_permuted.size(1), -1)
        #     key_layer = key_layer_reshaped * k_mask
        #     key_layer = self.transpose_for_scores(key_layer)
        #
        # if v_mask is not None:
        #     value_layer_permuted = value_layer.permute(0, 2, 1, 3)
        #     value_layer_reshaped = value_layer_permuted.view(value_layer_permuted.size(0), value_layer_permuted.size(1), -1)
        #     value_layer= value_layer_reshaped * v_mask
        #     value_layer = self.transpose_for_scores(value_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        key_layer, value_layer, attention_mask = self.prefix_tuning(key_layer, value_layer, attention_mask)

        # # TODO： 如果单独传qk   mask，在这儿分别 * mask
        # # 首先，使用 permute 重新排列维度，使得注意力头的维度放在序列长度之后
        # query_layer_permuted = query_layer.permute(0, 2, 1, 3)
        # query_layer_reshaped = query_layer_permuted.view(query_layer_permuted.size(0), query_layer_permuted.size(1), -1)
        # query_layer = torch.matmul(query_layer_reshaped, q_mask)
        # query_layer = self.transpose_for_scores(query_layer)
        # # print(f"query_layer.shape:{query_layer.shape}")
        #
        # key_layer_permuted = key_layer.permute(0, 2, 1, 3)
        # key_layer_reshaped = key_layer_permuted.view(key_layer_permuted.size(0), key_layer_permuted.size(1), -1)
        # key_layer = torch.matmul(key_layer_reshaped, k_mask)
        # key_layer = self.transpose_for_scores(key_layer)
        # # print(f"key_layer.shape:{key_layer.shape}")

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # TODO: 让 a_s直接成 12 * 768 的mask

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # TODO:value_layer进行修改
        # TODO
        # value_layer_permuted = value_layer.permute(0, 2, 1, 3)
        # value_layer_reshaped = value_layer_permuted.view(value_layer_permuted.size(0), value_layer_permuted.size(1), -1)
        # value_layer=torch.matmul(value_layer_reshaped , v_mask)
        # value_layer = self.transpose_for_scores(value_layer)
        # # print(f"value_layer.shape:{value_layer.shape}")

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

        # output = (context_layer, attention_probs) if output_attentions else (context_layer,)
        #
        # if self.is_decoder:
        #     outputs = output + (past_key_value,)
        #
        # attention_output = self.output(output[0], hidden_states, down_mask, up_mask, **kwargs)
        # outputs = (attention_output,) + output[1:]
        # return outputs

# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
class MyRobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, location_key: Optional[str] = None,args=None):
        super().__init__()
        self.self = OurRobertaSelfAttention(config, position_embedding_type=position_embedding_type, location_key=location_key)
        self.output = MyRobertaSelfOutput(config,args)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        q_mask=None,
        k_mask=None,
        v_mask=None,
        sigma_q_before=None,
        sigma_k_before=None,
        sigma_q_curr=None,
        sigma_k_curr=None,
        befor_q_mask=None,
        befor_k_mask=None,
        curr_q_mask=None,
        curr_k_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        down_mask=None,
        up_mask=None,
        **kwargs

    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            q_mask,
            k_mask,
            v_mask,
            sigma_q_before,
            sigma_k_before,
            sigma_q_curr,
            sigma_k_curr,
            befor_q_mask,
            befor_k_mask,
            curr_q_mask,
            curr_k_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states,down_mask, up_mask, **kwargs)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

# Copied from transformers.models.bert.modeling_bert.BertOutput
class MyRobertaOutput(BertOutputAdaptersMixin, nn.Module):
    def __init__(self, config, args):
        BertOutputAdaptersMixin.__init__(self, args)
        nn.Module.__init__(self)

        self.config = config

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_adapter_modules()

    def forward(self, hidden_states, input_tensor, output_mask, down_mask, up_mask, **kwargs):
        if output_mask is not None:
            hidden_states = self.dense(hidden_states) * output_mask
        else:
            hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter_layer_forward(hidden_states, input_tensor, 'encoder', down_mask, up_mask,
                                                   self.LayerNorm)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class MyRobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, intermediate_mask):
        if intermediate_mask is not None:
            hidden_states = self.dense(hidden_states) * intermediate_mask
        else:
            hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class MyRobertaLayer(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MyRobertaAttention(config, location_key="self", args=args)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = MyRobertaAttention(config, position_embedding_type="absolute", location_key="cross",
                                                     args=args)
        self.intermediate = MyRobertaIntermediate(config)
        self.output = MyRobertaOutput(config, args)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            q_mask=None,
            k_mask=None,
            v_mask=None,
            sigma_q_before=None,
            sigma_k_before=None,
            sigma_q_curr=None,
            sigma_k_curr=None,
            befor_q_mask=None,
            befor_k_mask=None,
            curr_q_mask=None,
            curr_k_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            output_mask=None,
            intermediate_mask=None,
            down_mask=None,
            up_mask=None,
            **kwargs

    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            q_mask,
            k_mask,
            v_mask,
            sigma_q_before,
            sigma_k_before,
            sigma_q_curr,
            sigma_k_curr,
            befor_q_mask,
            befor_k_mask,
            curr_q_mask,
            curr_k_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            down_mask=down_mask,
            up_mask=up_mask
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                q_mask,
                k_mask,
                v_mask,
                sigma_q_before,
                sigma_k_before,
                sigma_q_curr,
                sigma_k_curr,
                befor_q_mask,
                befor_k_mask,
                curr_q_mask,
                curr_k_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output,
            output_mask=output_mask, intermediate_mask=intermediate_mask,
            down_mask=down_mask, up_mask=up_mask, **kwargs
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output, output_mask, intermediate_mask, down_mask, up_mask, **kwargs):
        intermediate_output = self.intermediate(attention_output, intermediate_mask)
        layer_output = self.output(intermediate_output, attention_output, output_mask, down_mask, up_mask, **kwargs)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class MyRobertaEncoder(nn.Module):  # while we can use EobertaEncoder as father, it is risky that we may miss somehting
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        self.layer = nn.ModuleList([MyRobertaLayer(config, args) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            q_mask=None,
            k_mask=None,
            v_mask=None,
            sigma_q_before=None,
            sigma_k_before=None,
            sigma_q_curr=None,
            sigma_k_curr=None,
            befor_q_mask=None,
            befor_k_mask=None,
            curr_q_mask=None,
            curr_k_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            output_mask=None,
            intermediate_mask=None,
            down_mask=None,
            up_mask=None,
            **kwargs
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_q_mask = q_mask[i] if q_mask is not None else None
            layer_k_mask = k_mask[i] if k_mask is not None else None
            layer_v_mask = v_mask[i] if v_mask is not None else None
            #TODO
            layer_sigma_q_before = sigma_q_before[i] if sigma_q_before is not None else None
            layer_sigma_k_before = sigma_k_before[i] if sigma_k_before is not None else None
            layer_sigma_q_curr = sigma_q_curr[i] if sigma_q_curr is not None else None
            layer_sigma_k_curr = sigma_k_curr[i] if sigma_k_curr is not None else None
            layer_befor_q_mask = befor_q_mask[i] if befor_q_mask is not None else None
            layer_befor_k_mask = befor_k_mask[i] if befor_k_mask is not None else None
            layer_curr_q_mask = curr_q_mask[i] if curr_q_mask is not None else None
            layer_curr_k_mask = curr_k_mask[i] if curr_k_mask is not None else None

            layer_output_mask = output_mask[i] if output_mask is not None else None
            layer_intermediate_mask = intermediate_mask[i] if intermediate_mask is not None else None
            layer_down_mask = down_mask[i] if down_mask is not None else None
            layer_up_mask = up_mask[i] if up_mask is not None else None

            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    layer_q_mask,
                    layer_k_mask,
                    layer_v_mask,
                    layer_sigma_q_before,
                    layer_sigma_k_before,
                    layer_sigma_q_curr,
                    layer_sigma_k_curr,
                    layer_befor_q_mask,
                    layer_befor_k_mask,
                    layer_curr_q_mask,
                    layer_curr_k_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    layer_output_mask,
                    layer_intermediate_mask,
                    layer_down_mask,
                    layer_up_mask,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]
            (attention_mask,) = adjust_tensors_for_parallel(hidden_states, attention_mask)

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


ROBERTA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`RobertaTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class MyRobertaModel(BertModelAdaptersMixin, RobertaPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True, args=None):
        super().__init__(config)
        self.config = config
        self.args = args

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = MyRobertaEncoder(config, args)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self._init_adapter_modules()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    @ForwardContext.wrap
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            q_mask=None,
            k_mask=None,
            v_mask=None,
            sigma_q_before=None,
            sigma_k_before=None,
            sigma_q_curr=None,
            sigma_k_curr=None,
            befor_q_mask=None,
            befor_k_mask=None,
            curr_q_mask=None,
            curr_k_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            output_mask=None,
            intermediate_mask=None,
            down_mask=None,
            up_mask=None,
            **kwargs
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # print(f"MyRobertaModel head mask before:{head_mask.shape}")
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # print(f"MyRobertaModel head mask after:{head_mask.shape}")

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        embedding_output = self.invertible_adapters_forward(embedding_output)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            q_mask=q_mask,
            k_mask=k_mask,
            v_mask=v_mask,
            sigma_q_before=sigma_q_before,
            sigma_k_before=sigma_k_before,
            sigma_q_curr=sigma_q_curr,
            sigma_k_curr=sigma_k_curr,
            befor_q_mask=befor_q_mask,
            befor_k_mask=befor_k_mask,
            curr_q_mask=curr_q_mask,
            curr_k_mask=curr_k_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_mask=output_mask,
            intermediate_mask=intermediate_mask,
            down_mask=down_mask,
            up_mask=up_mask,
            **kwargs
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class MyRobertaForSequenceClassification(ModelWithHeadsAdaptersMixin, RobertaPreTrainedModel):
    # TODO: we also need to train MLM for representation learning1
    # TODO: finetune
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = MyRobertaModel(config, add_pooling_layer=False, args=args)
        self.args = args

        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            q_mask=None,
            k_mask=None,
            v_mask=None,
            sigma_q_before=None,
            sigma_k_before=None,
            sigma_q_curr=None,
            sigma_k_curr=None,
            befor_q_mask=None,
            befor_k_mask=None,
            curr_q_mask=None,
            curr_k_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            output_mask=None,
            intermediate_mask=None,
            down_mask=None,
            up_mask=None,
            my_loss=None,
            only_return_output=None
    ):

        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            q_mask=q_mask,
            k_mask=k_mask,
            v_mask=v_mask,
            sigma_q_before=sigma_q_before,
            sigma_k_before=sigma_k_before,
            sigma_q_curr=sigma_q_curr,
            sigma_k_curr=sigma_k_curr,
            befor_q_mask=befor_q_mask,
            befor_k_mask=befor_k_mask,
            curr_q_mask=curr_q_mask,
            curr_k_mask=curr_k_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_mask=output_mask,
            intermediate_mask=intermediate_mask,
            return_dict=return_dict,
            down_mask=down_mask,
            up_mask=up_mask
        )

        if only_return_output: return outputs

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if my_loss is not None:
            loss += my_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # SequenceClassifierOutput
        return MyOutput(
            loss=loss,
            logits=logits,
            sequence_output=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """, ROBERTA_START_DOCSTRING)
#TODO
class CustomMaskedLMOutput(MaskedLMOutput):
    def __init__(self, loss=None, logits=None, hidden_states=None, attentions=None, entropy_loss=None):
        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
        self.entropy_loss = entropy_loss
class MyRobertaForMaskedLM(ModelWithHeadsAdaptersMixin, RobertaPreTrainedModel):
    #TODO:posttrain
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, args):
        super().__init__(config)

        self.roberta = MyRobertaModel(config, add_pooling_layer=False, args=args)
        self.lm_head = RobertaLMHead(config)
        self.args = args

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        # Initialize weights and apply final processing

        if 'adapter_classic' in args.baseline:
            self.self_attns = nn.ModuleList()
            for t in range(args.ntasks):
                self.self_attns.append(utils.model.Self_Attn(t + 1))

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            q_mask=None,
            k_mask=None,
            v_mask=None,
            sigma_q_before=None,
            sigma_k_before=None,
            sigma_q_curr=None,
            sigma_k_curr=None,
            befor_q_mask=None,
            befor_k_mask=None,
            curr_q_mask=None,
            curr_k_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            adapter_names=None,
            output_mask=None,
            intermediate_mask=None,
            down_mask=None,
            up_mask=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            q_mask=q_mask,
            k_mask=k_mask,
            v_mask=v_mask,
            sigma_q_before=sigma_q_before,
            sigma_k_before=sigma_k_before,
            sigma_q_curr=sigma_q_curr,
            sigma_k_curr=sigma_k_curr,
            befor_q_mask=befor_q_mask,
            befor_k_mask=befor_k_mask,
            curr_q_mask=curr_q_mask,
            curr_k_mask=curr_k_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adapter_names=adapter_names,
            output_mask=output_mask,
            intermediate_mask=intermediate_mask,
            down_mask=down_mask,
            up_mask=up_mask
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(
            sequence_output,
            inv_lang_adapter=self.roberta.get_invertible_adapter(),
        )

        masked_lm_loss = None
        if labels is not None:
            # print(f"prediction_scores:{prediction_scores.view(-1, self.config.vocab_size)}")
            # print(f"labels:{labels}")
            prediction_scores = prediction_scores.float()
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)).float()

            # TODO：添加熵损失
            # probs = F.softmax(prediction_scores, dim=-1)
            # log_probs = F.log_softmax(prediction_scores, dim=-1)
            # entropy_loss = -(probs * log_probs).sum(dim=-1).mean()

            # entropy_per_token = -(probs * log_probs).sum(dim=-1)  # 每个token的熵
            # entropy_loss = entropy_per_token.mean()  # 平均熵
            # # 加权损失
            # total_loss = masked_lm_loss + entropy_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # return MaskedLMOutput(
        return CustomMaskedLMOutput(  # 使用自定义的输出类
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
            # entropy_loss=entropy_loss,  # 新增的字段
        )


def apply_chunking_to_forward(
        forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors, output_mask=None,
        intermediate_mask=None, **kwargs
) -> torch.Tensor:
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    forward_fn_params = inspect.signature(forward_fn).parameters
    num_args_in_forward_chunk_fn = len(forward_fn_params)
    # subtract one for kwargs
    if "kwargs" in forward_fn_params:
        num_args_in_forward_chunk_fn -= 1
    # I manually add something the forward chunk
    # if num_args_in_forward_chunk_fn != len(input_tensors):
    #     raise ValueError(
    #         f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
    #         "tensors are given"
    #     )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors, output_mask=output_mask, intermediate_mask=intermediate_mask, **kwargs)


class MyOutput(ModelOutput):
    loss = None
    logits = None
    sequence_output = None
    hidden_states = None
    attentions = None