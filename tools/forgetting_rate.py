import json
import os.path
import random

import jsonlines
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


# seed = ['2021','111','222','333','444']
# seed = ['111','222','444']
# seed = ['2021','111','222','333','444']
# seed = ['2021','111','222']
seed = ['2021']

f1 = "/home/longjing/zlz/ContinualLM-main-ori/ContinualLM-main_kv_grad/output_ori_0.1_e4_256_2500_random_50/seq0/640000samples/das/"
acc= "/home/longjing/zlz/ContinualLM-main-ori/ContinualLM-main_kv_grad/output_ori_0.1_e4_256_2500_random_50/seq0/640000samples/das/"
for s in seed:
    if f1 is None:
        f1 = np.loadtxt('progressive_f1_'+s)
        acc = np.loadtxt('progressive_acc_'+s)
    else:
        new_path_f1=f1+'progressive_f1_'+s
        new_path_acc=acc+'progressive_acc_'+s
        
        f1 = np.loadtxt(new_path_f1)
        acc = np.loadtxt(new_path_acc)


avg_f1 = f1 / len(seed) * 1.0
avg_acc = acc / len(seed) * 1.0

print('avg_f1: ',avg_f1)
print('avg_acc: ',avg_acc)


fr_f1 = []
fr_acc = []

ntasks = len(avg_f1)
for i in range(ntasks-1):

    fr_f1.append(avg_f1[i][i] - avg_f1[ntasks-1][i])
    fr_acc.append(avg_acc[i][i] - avg_acc[ntasks-1][i])

print('fr_f1: ',fr_f1)
print('fr_acc: ',fr_acc)

fr_f1 = np.mean(fr_f1)
fr_acc = np.mean(fr_acc)

print('fr_f1: ',fr_f1)
print('fr_acc: ',fr_acc)
