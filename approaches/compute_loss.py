import logging
import torch
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import utils
from copy import deepcopy

# def compute(self,model,batch,head_impt,intermediate_impt,output_impt,self_fisher,mask_pre,train_loader,step,accelerator):
#
#     self.args.s = (self.args.smax - 1 / self.args.smax) * step / len(train_loader) + 1 / self.args.smax # Only for HAT based model
#
#     if 'ewc' in self.args.baseline:
#         outputs = model(batch, self_fisher=self_fisher)
#     elif 'adapter_hat' in self.args.baseline \
#             or 'adapter_bcl' in self.args.baseline \
#             or 'adapter_classic' in self.args.baseline:
#         masks = self.mask(model, accelerator, self.args)
#         outputs = model(batch, masks=masks, mask_pre=mask_pre)
#     elif 'transformer_hat' in self.args.baseline:
#         model_ori = accelerator.unwrap_model(model)
#         head_importance, intermediate_importance, output_importance = model_ori.model.transformer_mask()
#         masks = self.mask(model, accelerator, self.args)  # need mask
#         outputs = model(batch, head_mask=head_importance,
#                         intermediate_mask=intermediate_importance, output_mask=output_importance,
#                         masks=masks, mask_pre=mask_pre)
#
#     elif 'dga' in self.args.baseline or 'das' in self.args.baseline:
#         outputs = model(batch,
#                         head_mask=head_impt,
#                         intermediate_mask=intermediate_impt,
#                         output_mask=output_impt)
#                         # q_mask=q_impt,
#                         # k_mask=k_impt,
#                         # v_mask=v_impt) # TODO: 多传一个qk_impt，维度12 * 768
#     else:
#         outputs = model(batch)
#     return self, model, outputs

# def compute(self,model,batch,q_impt,k_impt,self_fisher,mask_pre,train_loader,step,accelerator):
def compute(self,model,batch,
            # sigma_q_before, sigma_k_before, sigma_q_curr, sigma_k_curr,
            befor_q_impt, befor_k_impt,
            # curr_q_impt, curr_k_impt,
            self_fisher,mask_pre,train_loader,step,accelerator):

    self.args.s = (self.args.smax - 1 / self.args.smax) * step / len(train_loader) + 1 / self.args.smax # Only for HAT based model

    if 'ewc' in self.args.baseline:
        outputs = model(batch, self_fisher=self_fisher)
    elif 'adapter_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        masks = self.mask(model, accelerator, self.args)
        outputs = model(batch, masks=masks, mask_pre=mask_pre)
    elif 'transformer_hat' in self.args.baseline:
        model_ori = accelerator.unwrap_model(model)
        head_importance, intermediate_importance, output_importance = model_ori.model.transformer_mask()
        masks = self.mask(model, accelerator, self.args)  # need mask
        outputs = model(batch, head_mask=head_importance,
                        intermediate_mask=intermediate_importance, output_mask=output_importance,
                        masks=masks, mask_pre=mask_pre)

    elif 'dga' in self.args.baseline or 'das' in self.args.baseline:
        # print("进入了这里的损失函数")
        # print(model)
        # q_mask=q_impt
        # k_mask=k_impt
        # sigma_q_before=sigma_q_before
        # sigma_k_before=sigma_k_before
        # sigma_q_curr=sigma_q_curr
        # sigma_k_curr=sigma_k_curr
        befor_q_mask=befor_q_impt
        befor_k_mask=befor_k_impt
        # curr_q_mask=curr_q_impt
        # curr_k_mask=curr_k_impt
        # v_mask=v_impt
        outputs = model(batch,
                        # sigma_q_before=sigma_q_before,
                        # sigma_k_before=sigma_k_before,
                        # sigma_q_curr=sigma_q_curr,
                        # sigma_k_curr=sigma_k_curr,
                        befor_q_mask=befor_q_mask,
                        befor_k_mask=befor_k_mask,
                        # curr_q_mask=curr_q_mask,
                        # curr_k_mask=curr_k_mask
                        # q_mask=q_mask,
                        # k_mask=k_mask,
                        # head_mask=head_mask,
                        # intermediate_mask=intermediate_impt,
                        # output_mask=output_impt
                       ) # TODO: 多传一个qk_impt，维度12 * 768

    else:
        outputs = model(batch)
    return self, model, outputs
