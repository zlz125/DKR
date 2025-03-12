import torch
import torch.nn as nn
import numpy as np
import scipy.stats
from tqdm import tqdm
from scipy.linalg import svd
# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        #self为参数矩阵，q,k,v分别为[4096,4096]
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        # self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name
    
    
    #inp的形状为[164,768],out的形状为[128, 164, 768]
    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            #列的维度不变，将前两个维度合并
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        
        inp = inp.type(torch.float32)
        
        device = inp.device
        
        # #基于SUV的方法
        # # 计算 X 的 SVD 分解
        # U, s, Vt = svd(inp.to("cpu"), full_matrices=False)

        # # 设置阈值
        # threshold = 1e-5

        # # 创建对角矩阵 Sigma 的逆的近似
        # Sigma_inv = np.diag(1.0 / s)

        # # 忽略小于阈值的奇异值
        # Sigma_inv[s < threshold] = 0

        # # 计算 (X^T X)^(-1) 的近似
        # XXT_inv_approx = U @ Sigma_inv @ Sigma_inv @ U.T
        
        # # 计算逆矩阵的对角线
        # diagonal_inv_inp = np.diag(XXT_inv_approx)
        # result=torch.from_numpy(diagonal_inv_inp)
        # self.scaler_row+=result.to(device)/self.nsamples
        
        #计算每列的平均值
        #self.scaler_row += torch.mean(inp, dim=1) / self.nsamples
             
       
        #计算每列的平均值,列平均的方法 
        column_means = torch.mean(inp, dim=0) 
        
        # 计算 inp 与 mean 之间的欧几里得距离
        distances = np.linalg.norm(inp.to("cpu") - column_means.to("cpu"), axis=1)
        result=torch.from_numpy(distances)
        self.scaler_row += result.to(device)  / self.nsamples
       
        # #print(column_means.shape)
       
        # # 对每行与平均行向量之间的JS散度，JS散度方法
        # result=[]
        # for raw in tqdm(range(inp.shape[0])):
        #     #计算中间值
        #     #print("输入形状")
        #     #print(inp[raw].shape)
        #     M=(inp[raw]+column_means)/2
        #     js_distance = 0.5*scipy.stats.entropy(inp[raw].to("cpu"), M.to("cpu"), base=2)+0.5*scipy.stats.entropy(column_means.to("cpu"), M.to("cpu"), base=2)
        #     result.append(js_distance)
        # result=torch.Tensor(result)    
        
        # # 将每行的JS散度累加到self.scaler_row中
        # self.scaler_row += result.to(device) / self.nsamples
        
        #基于逆矩阵计算重要度
        
        # 计算逆矩阵
        # X=inp @ inp.t()
        # inp_inv = np.linalg.inv(X.to("cpu"))

        # # 计算逆矩阵的对角线
        # diagonal_inv_inp = np.diag(inp_inv)
        # result=torch.from_numpy(diagonal_inv_inp)
        # self.scaler_row+=result.to(device)/self.nsamples
        
        #基于范数的方法
        #self.scaler_row += torch.norm(inp, p=1, dim=1) ** 2  / self.nsamples
        #self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        