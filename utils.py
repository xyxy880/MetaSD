import cv2
import numpy as np
import math
from time import strftime,localtime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import torch
import csv
import os

def task_weight(T):
    epsilon = 1e-8 #防止分母为0
    all = sum(T)+epsilon#所有任务总的loss
    n=len(T)#任务总数
    soft = [min(t*n/all/2 + 0.27, 0.99) for t in T]#将值映射到0.3到0.99
    loss_w = [-pow(s, 3) * math.log2(1 - s) for s in soft]
    for i in range(n):
        if loss_w[i]<1:
            loss_w[i]=1.0
    return loss_w

def task_weight1(T):
    epsilon = 1e-8 #防止分母为0
    all = sum(T)+epsilon#所有任务总的loss
    n=len(T)#任务总数
    soft = [min(t*n/all/2 + 0.27, torch.tensor(0.99)) for t in T]#将值映射到0.3到0.99
    loss_w = [-pow(s, 3) * math.log2(1 - s) for s in soft]
    for i in range(n):
        if loss_w[i]<1:
            loss_w[i]=torch.tensor(1.0)
    return loss_w

def iter_weights(iter,step):
    loss_weights = torch.ones(iter) * (1.0 / iter)
    decay_rate = 1.0 / iter / (10000 / 3)
    min_value = 0.03 / iter
    loss_weights_pre = torch.clamp(loss_weights[:-1] - (step * decay_rate), min=min_value)

    loss_weight_cur = torch.clamp(loss_weights[-1] + (step * (iter - 1) * decay_rate),
                                  max=1.0 - ((iter - 1) * min_value))
    loss_weights = torch.cat([loss_weights_pre, loss_weight_cur.unsqueeze(0)], dim=0)
    return loss_weights

def get_start_lr(lr,hr):
    sr = cv2.resize(lr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    p=psnr(sr, hr, data_range=1.0)

    if p<=40:
        lr=6e-4
    else:
        lr=1e-4

    return lr

def mean_psrcm(imgs1,imgs2,data_range,path,MAX_MIN_DIR):
    def inverse_max_min_normalize(img, max_value, min_value):
        return np.multiply(img, (max_value - min_value)) + min_value

    def ACC(x, y):
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        pearson = np.corrcoef(x, y)[0, 1]
        return pearson

    def MBA(x, y):
        # 计算平均值
        mba = np.mean(x) - np.mean(y)
        return mba

    def get_max_min_dict(path):
        max_min = {}
        with open(MAX_MIN_DIR, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = ''
                maxmin = []
                for key, value in row.items():
                    if key == 'name':
                        name = value
                    else:
                        maxmin.append(value)
                max_min[name] = maxmin
        return np.float(max_min[path][0]), np.float(max_min[path][1])

    #imgs1_shape:B*H*W
    PP=[]
    SS=[]
    RR=[]
    CC=[]
    MM=[]
    for img1,img2 in zip(imgs1,imgs2):

        # max_value,min_value=get_max_min_dict(path)
        P=psnr(img1,img2,data_range=data_range)
        S=ssim(img1,img2,data_range=data_range)

        # img1=inverse_max_min_normalize(img1,max_value,min_value)
        # img2=inverse_max_min_normalize(img2,max_value,min_value)
        # P=psnr(img1,img2,data_range=max_value-min_value)
        # S=ssim(img1,img2,data_range=max_value-min_value)
        C = ACC(img1, img2)
        R = np.sqrt(mse(img1, img2))

        M=MBA(img1,img2)

        PP.append(P)
        SS.append(S)
        RR.append(R)
        CC.append(C)
        MM.append(M)


    print(f'{path}:',len(PP))

    return np.mean(PP),np.mean(SS),np.mean(RR),np.mean(CC),np.mean(MM)

def save(LR,SR,HR,dir,name):
    os.makedirs(f'./result/{dir}',exist_ok=True)
    np.save(f'./result/{dir}/{name}_LR', LR)
    np.save(f'./result/{dir}/{name}_SR', SR)
    np.save(f'./result/{dir}/{name}_HR', HR)


def print_time():
    print('Time: ', strftime('%b-%d %H:%M:%S', localtime()))

if __name__=='__main__':
    # print(iter_weights(3, 234))
    print(task_weight1([1,1,2,1,3,4,1]))