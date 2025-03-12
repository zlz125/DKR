import torch.nn.functional as F
import logging
import math
import os
import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from networks import prompt
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
import numpy as np
# from dataloader.data import get_dataset
from dataloader.data import get_dataset,dataset_class_num#获取无标注的语料和分类数
from torch.utils.data import DataLoader
import random
from accelerate import Accelerator

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from utils import utils

class Acc(object):

    def __init__(self,args):
        super().__init__()
        self.args=args
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        return

    def acc_sim(self):
        self.args.sequence_file = 'posttrain'
        self.args.base_model_name_or_path = "roberta-base"

        sequence_path = f'./sequences/{self.args.sequence_file}'
        print(f'sequence_path:{sequence_path}')
        with open(sequence_path, 'r') as f:
            datas = f.readlines()[self.args.idrandom]
            data = datas.split()

        posttrain2endtask = {"pubmed_unsup": "chemprot_sup", "phone_unsup": "phone_sup", "ai_unsup": "scierc_sup",
                             "camera_unsup": "camera_sup", "acl_unsup": "aclarc_sup",
                             "restaurant_unsup": "restaurant_sup"}

        output = f'{self.args.base_dir}/seq{self.args.idrandom}/{self.args.max_samples}samples/{self.args.baseline}/{data[self.args.pt_task]}_roberta/'
        # ckpt = f'{self.args.base_dir}/seq{self.args.idrandom}/{self.args.max_samples}samples/{self.args.baseline}/{data[self.args.pt_task]}_roberta/'

        self.args.acc_dataset_name = posttrain2endtask[data[self.args.pt_task]]
        # self.args.model_name_or_path = ckpt

        self.args.task = self.args.pt_task

        print(f'Output directory: {self.args.output_dir}')
        print(f'Dataset: {self.args.acc_dataset_name}')
        print(f'Pretrained model: {self.args.model_name_or_path}')

        if self.args.acc_dataset_name in ['aclarc_sup']:
            self.args.epoch = 10
        elif self.args.acc_dataset_name in ["hoc_multi", "scierc_sup", "covidintent_sup", 'restaurant_sup', "laptop_sup"]:
            self.args.epoch = 5
        elif self.args.acc_dataset_name in ['phone_sup', "camera_sup"]:
            self.args.epoch = 15
        elif self.args.acc_dataset_name in ['chemprot_sup', 'rct_sample_sup', 'electric_sup', 'hyperpartisan_sup']:
            self.args.epoch = 10

        self.args.s = self.args.smax

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        accelerator = Accelerator()
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state)
        logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

        # if args.log_dir is not None:
        #     handler = logging.FileHandler(args.log_dir)
        #     handler.setLevel(logging.INFO)
        #     logger.addHandler(handler)
        #
        # console = logging.StreamHandler()
        # console.setLevel(logging.INFO)
        # logger.addHandler(console)

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # if use multilabel-classification datasets:
        if 'multi' in self.args.acc_dataset_name:
            self.args.problem_type = 'multi_label_classification'

        # Handle the repository creation
        # if accelerator.is_main_process:
        #     if args.output_dir is not None:
        #         os.makedirs(args.output_dir, exist_ok=True)
        # accelerator.wait_for_everyone()

        # Get the datasets and process the data.
        tokenizer = RobertaTokenizer.from_pretrained(self.args.model_name_or_path)
        # tokenizer = RobertaTokenizer.from_pretrained("/home/longjing/zlz/ContinualLM-main/roberta-base")
        self.args.tokenizer = tokenizer

        max_length = self.args.max_seq_length

        logger.info('==> Preparing data..')

        datasets = get_dataset(self.args.acc_dataset_name, tokenizer=tokenizer, args=self.args)
        print(f'Dataset: {self.args.acc_dataset_name}')

        print(f'Size of training set: {len(datasets["train"])}')
        print(f'Size of testing set: {len(datasets["test"])}')

        train_dataset = datasets['train']
        test_dataset = datasets['test']

        test_dataset = test_dataset.map(
            lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                                 num_workers=8)

        train_dataset = train_dataset.map(
            lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False,
                                  num_workers=8)  # consider batch size

        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}. Decode to: {tokenizer.decode(train_dataset[index]['input_ids'])}")

        if self.args.finetune_class_num is None:
            self.args.finetune_class_num  = dataset_class_num[self.args.acc_dataset_name]
        print(f"class_num:{self.args.finetune_class_num }")

        # Declare the model and set the training parameters.
        logger.info('==> Building model..')

        model = utils.model.lookfor_model_finetune(self.args)

        # 调用auc计算正确率
        accuracy, auc_score = self.train(model, accelerator, train_loader, test_loader)
        return accuracy, auc_score

    # TODO: Multiple-GPU supprt

    def train(self,model,accelerator,train_loader, test_loader):

        # Set the optimizer 设置优化器
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                          weight_decay=self.args.weight_decay)

        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.finetune_gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epoch * num_update_steps_per_epoch
        else:
            self.args.epoch = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Prepare everything with the accelerator
        model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

        logger.info("***** Running training *****")
        logger.info( f"Pretrained Model = {self.args.model_name_or_path},  Dataset name = {self.args.acc_dataset_name}, seed = {self.args.seed}")

        summary_path = f'{self.args.output_dir}../{self.args.acc_dataset_name}_finetune_summary'
        # summary_path = f'{self.args.output_dir}'
        print(f'summary_path: {summary_path}')

        for epoch in range(self.args.epoch):
            print("Epoch {} started".format(epoch))
            train_acc, training_loss = self.train_epoch(model, optimizer, train_loader, accelerator, lr_scheduler)
            print("train acc = {:.4f}, training loss = {:.4f}".format(train_acc, training_loss))

        #这里的计算方法用auc替换

        #micro_f1, macro_f1, acc, test_loss = self.eval(model, test_loader, accelerator)
        # Calculate AUC instead of F1
        accuracy, auc_score, test_loss, fpr, tpr = self.eval_roc_2(model, test_loader, accelerator)

        # task_name = test_loader.dataset.task_name
        print(f"AUC score on {self.args.acc_dataset_name} test set: {auc_score:.4f}")
        print(f"Accuracy on {self.args.acc_dataset_name} test set: {accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        # Save ROC curve plot
        # self.save_roc_plot(fpr, tpr, auc_score, task_name)

        return accuracy,auc_score


    def train_epoch(self,model, optimizer, dataloader, accelerator, lr_scheduler):
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        model.train()
        train_acc = 0.0
        training_loss = 0.0
        total_num = 0.0
        for batch, inputs in enumerate(dataloader):
            if 'transformer_hat' in self.args.baseline:
                model_ori = accelerator.unwrap_model(model)
                #TODO
                # head_importance, intermediate_importance, output_importance = model_ori.transformer_mask()
                # res = model.model(**inputs, head_mask=head_importance, intermediate_mask=intermediate_importance,
                #                 output_mask=output_importance)
                
                head_importance, intermediate_importance, output_importance = model_ori.transformer_mask()
                res = model.model(**inputs, head_mask=head_importance, intermediate_mask=intermediate_importance,
                                output_mask=output_importance)
            else:
                #TODO
                # q_mask = torch.ones(12,768,768).cuda()
                # k_mask = torch.ones(12,768,768).cuda()
                # v_mask = torch.ones(12,768,768).cuda()
                res = model.model(**inputs)

            outp = res.logits
            loss = res.loss
            optimizer.zero_grad()
            accelerator.backward(loss)

            # for n,p in accelerator.unwrap_model(model).named_parameters():
            #     if p.grad is not None:
            #         print('n,p： ',n)

            optimizer.step()
            lr_scheduler.step()

            pred = outp.max(1)[1]

            predictions = accelerator.gather(pred)
            references = accelerator.gather(inputs['labels'])


            train_acc += (references == predictions).sum().item()
            training_loss += loss.item()
            total_num += references.size(0)

            progress_bar.update(1)
            # break
        return train_acc / total_num, training_loss / total_num
    #     return micro_f1, macro_f1, accuracy,total_loss/total_num

    def compute_macro_roc(self,label_list, prediction_probs):
        num_classes = len(set(label_list))
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        # 确保 prediction_probs 是 NumPy 数组
        prediction_probs = np.array(prediction_probs)
        # 确保 label_list 也是 NumPy 数组
        label_list = np.array(label_list)

        for i in range(num_classes):
            # 针对每个类别计算ROC曲线
            # 将 label_list 转换成二分类标签
            binary_labels = (label_list == i).astype(int)
            # 获取对应类别的预测概率
            class_probs = prediction_probs[:, i]
            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            mean_tpr += np.interp(mean_fpr, fpr, tpr)

        mean_tpr /= num_classes
        macro_auc = auc(mean_fpr, mean_tpr)
        return mean_fpr, mean_tpr, macro_auc
    def eval_roc_2(self, model, dataloader, accelerator):
        model.eval()
        label_list = []
        prediction_list = []
        prediction_probs = []
        total_loss = 0
        total_num = 0
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)

        with torch.no_grad():
            for batch, inputs in enumerate(dataloader):
                input_ids = inputs['input_ids']
                res = model.model(**inputs, return_dict=True)
                real_b = input_ids.size(0)
                loss = res.loss
                outp = res.logits
                
                # 将logits转换为概率
                probs = F.softmax(outp, dim=1)
                
                # 确定预测的类别
                if self.args.problem_type != 'multi_label_classification':
                    pred = probs.max(1)[1]
                else:
                    pred = outp.sigmoid() > 0.5

                total_loss += loss.data.cpu().numpy().item() * real_b
                total_num += real_b

                # Gather predictions and labels
                predictions = accelerator.gather(pred)
                references = accelerator.gather(inputs['labels'])
                probabilities = accelerator.gather(probs)

                label_list += references.cpu().numpy().tolist()
                prediction_list += predictions.cpu().numpy().tolist()
                prediction_probs += probabilities.cpu().numpy().tolist()
                progress_bar.update(1)
        
        # 计算accuracy
        accuracy = sum([float(label_list[i] == prediction_list[i]) for i in range(len(label_list))]) * 1.0 / len(prediction_list)

        num_classes = len(set(label_list))
        # 如果是单分类问题，使用概率进行ROC曲线的绘制
        if num_classes<=2:
            fpr, tpr, _ = roc_curve(label_list, [pred[1] for pred in prediction_probs], pos_label=1)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
        else:
            # 对于多分类问题，计算每个类别的AUC-ROC分数
            # num_classes = len(set(label_list))
            auc_score = roc_auc_score(label_list, prediction_probs, average='macro', multi_class='ovo')

            for i in range(num_classes):
                # 针对每个类别，计算ROC曲线
                fpr, tpr, _ = roc_curve([1 if label == i else 0 for label in label_list],
                                        [pred[i] for pred in prediction_probs])
                plt.plot(fpr, tpr,linestyle='-', label=f'ROC curve of class {i+1} (area = {auc(fpr, tpr):.2f})')

            # 绘制macro-average ROC曲线
            # 计算宏平均ROC曲线
            macro_fpr, macro_tpr, auc_score = self.compute_macro_roc(label_list, prediction_probs)
            plt.plot(macro_fpr, macro_tpr,color='darkblue', marker='^',linestyle='', label=f'macro-average ROC curve (area = {auc_score:.2f})')
            
        # 绘制随机预测的对角线
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')

        # 添加网格线和浅灰背景
        plt.grid(True, which='both', linestyle='-', linewidth=1, alpha=0.5)  # 网格线样式设置
        plt.gca().set_facecolor('whitesmoke')

        # 设置图形标签和标题
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'The ROC curve of the dataset {self.args.acc_dataset_name} on the previous model')
        plt.legend(loc="lower right")
        
        # 保存图像
        file_name = f'roc_score{self.args.pt_task}.png'
        file_path = os.path.join(self.args.output_dir, file_name)
        plt.savefig(file_path)
        plt.show()
        plt.close()

        return accuracy, auc_score, total_loss / total_num, fpr, tpr
    
    
    def save_roc_plot(self, fpr, tpr, auc_score, task_name):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        
        # Ensure the directory exists
        # os.makedirs("/home/longjing/zlz/ContinualLM-main/output_qk_sigma_2021_2500/roc_picture", exist_ok=True)
        
        # Save the plot
        # plt.savefig(f"/home/qhx/ContinualLM-main_kv_grad_test/roc_picture/{task_name}_roc.png")
        # 检查文件是否存在
        if os.path.exists(self.args.output_dir):
            # 如果文件存在，则不保存图像
            print(f"File {self.args.output_dir} already exists. Skipping save.")
        else:
            # 如果文件不存在，则保存图像
            plt.savefig(self.args.output_dir)
            print(f"Image saved to {self.args.output_dir}")
        plt.close()
