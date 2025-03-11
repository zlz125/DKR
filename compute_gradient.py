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
from networks.baselines import ewc, hat, softmask, memory

# def compute(self,model,head_impt, intermediate_impt, output_impt,batch, loss,buffer,mask_back,outputs,epoch,step,accelerator):
#
#     if 'derpp' in self.args.baseline \
#             and not (buffer is None or buffer.is_empty()) \
#             and step % self.args.replay_freq == 0:
#
#         replay_batch = buffer.get_datadict(size=batch['input_ids'].shape[0])
#         replay_outputs = model(replay_batch)
#
#         loss += replay_outputs.loss * self.args.replay_beta
#         loss += self.mse(
#             replay_outputs.hidden_states[-1], replay_batch['logits']) * self.args.replay_alpha
#
#
#     if 'dga' in self.args.baseline or 'das' in self.args.baseline:
#         contrast_loss = outputs.contrast_loss
#         loss = loss + contrast_loss
#     if 'distill' in self.args.baseline:
#         distill_loss = outputs.distill_loss
#         loss = loss + distill_loss
#     if 'simcse' in self.args.baseline:
#         simcse_loss = outputs.simcse_loss
#         loss = loss + simcse_loss
#     if 'tacl' in self.args.baseline:
#         tacl_loss = outputs.tacl_loss
#         loss = loss + tacl_loss
#     if 'taco' in self.args.baseline:
#         taco_loss = outputs.taco_loss
#         loss = loss + taco_loss
#     if 'infoword' in self.args.baseline:
#         infoword_loss = outputs.infoword_loss
#         loss = loss + infoword_loss
#
#     loss = loss / self.args.gradient_accumulation_steps
#     accelerator.backward(loss)
#
#     if accelerator.is_main_process and epoch < 1 and step < 1:
#         for n, p in accelerator.unwrap_model(model).named_parameters():
#             if p.grad is not None:
#                 print(f'Gradient of param "{n}" with size {tuple(p.size())} detected')
#
#     if self.args.pt_task > 0 and \
#             ('adapter_hat' in self.args.baseline
#              or 'transformer_hat' in self.args.baseline
#              or 'adapter_bcl' in self.args.baseline
#              or 'adapter_classic' in self.args.baseline):
#         for n, p in model.named_parameters():
#             if n in mask_back and p.grad is not None:
#                 p.grad.data *= mask_back[n]
#
#     if 'adapter_hat' in self.args.baseline \
#             or 'transformer_hat' in self.args.baseline \
#             or 'adapter_bcl' in self.args.baseline \
#             or 'adapter_classic' in self.args.baseline:
#         # Compensate embedding gradients
#         for n, p in model.named_parameters():
#             if ('adapters.e' in n or 'model.e' in n) and p.grad is not None:
#                 num = torch.cosh(torch.clamp(self.args.s * p.data, -self.args.thres_cosh,
#                                              self.args.thres_cosh)) + 1
#                 den = torch.cosh(p.data) + 1
#                 p.grad.data *= self.args.smax / self.args.s * num / den
#
#     # we need this even for the first task
#     if 'dga' in self.args.baseline or 'das' in self.args.baseline:
#         softmask.soft_mask_gradient(model, head_impt, intermediate_impt, output_impt, accelerator, epoch, step,
#                                     self.args)
#
#     return model

# def compute(self,model,q_impt_combine, k_impt_combine,batch, loss,buffer,mask_back,outputs,epoch,step,accelerator):
def compute(self,model,
            # sigma_q_before, sigma_k_before, sigma_q_curr, sigma_k_curr,
            q_impt, k_impt,
            # curr_q_impt, curr_k_impt,
            batch, loss,buffer,mask_back,outputs,epoch,step,accelerator):

    if 'derpp' in self.args.baseline \
            and not (buffer is None or buffer.is_empty()) \
            and step % self.args.replay_freq == 0:

        replay_batch = buffer.get_datadict(size=batch['input_ids'].shape[0])
        replay_outputs = model(replay_batch)

        loss += replay_outputs.loss * self.args.replay_beta
        loss += self.mse(
            replay_outputs.hidden_states[-1], replay_batch['logits']) * self.args.replay_alpha


    if 'dga' in self.args.baseline or 'das' in self.args.baseline:
        contrast_loss = outputs.contrast_loss
        loss = loss + contrast_loss
        # loss=loss
    if 'distill' in self.args.baseline:
        distill_loss = outputs.distill_loss
        loss = loss + distill_loss
    if 'simcse' in self.args.baseline:
        simcse_loss = outputs.simcse_loss
        loss = loss + simcse_loss
    if 'tacl' in self.args.baseline:
        tacl_loss = outputs.tacl_loss
        loss = loss + tacl_loss
    if 'taco' in self.args.baseline:
        taco_loss = outputs.taco_loss
        loss = loss + taco_loss
    if 'infoword' in self.args.baseline:
        infoword_loss = outputs.infoword_loss
        loss = loss + infoword_loss

    loss = loss / self.args.gradient_accumulation_steps
    accelerator.backward(loss)

    # mask=model.model.roberta.encoder.layer[0].attention.self.query.weight.grad
    # for lay in range(12):
    #     q_grad[lay]+=torch.abs(model.model.roberta.encoder.layer[lay].attention.self.query.weight.grad)
    #     k_grad[lay]+=torch.abs(model.model.roberta.encoder.layer[lay].attention.self.key.weight.grad)
    # head_grad+=torch.abs(head_mask.grad.detach())

    # q_grad += torch.abs(q_mask.grad.detach())
    # k_grad += torch.abs(k_mask.grad.detach())

    if accelerator.is_main_process and epoch < 1 and step < 1:
        for n, p in accelerator.unwrap_model(model).named_parameters():
            if p.grad is not None:
                # print(f"n:{n.grad}")
                print(f'Gradient of param "{n}" with size {tuple(p.size())} detected')

    if self.args.pt_task > 0 and \
            ('adapter_hat' in self.args.baseline
             or 'transformer_hat' in self.args.baseline
             or 'adapter_bcl' in self.args.baseline
             or 'adapter_classic' in self.args.baseline):
        for n, p in model.named_parameters():
            if n in mask_back and p.grad is not None:
                p.grad.data *= mask_back[n]

    if 'adapter_hat' in self.args.baseline \
            or 'transformer_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        # Compensate embedding gradients
        for n, p in model.named_parameters():
            if ('adapters.e' in n or 'model.e' in n) and p.grad is not None:
                num = torch.cosh(torch.clamp(self.args.s * p.data, -self.args.thres_cosh,
                                             self.args.thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= self.args.smax / self.args.s * num / den

    # we need this even for the first task
    if 'dga' in self.args.baseline or 'das' in self.args.baseline:
        softmask.soft_mask_gradient(model,
                                    # share_q_impt, uniq_q_impt, share_k_impt, uniq_k_impt,
                                    # sigma_q_before, sigma_k_before, sigma_q_curr, sigma_k_curr,
                                    q_impt, k_impt,
                                    # intermediate_impt, output_impt,
                                    # q_mask,k_mask,
                                    accelerator, epoch, step,self.args)

    return model