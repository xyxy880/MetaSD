import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as psnr_rgb
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import mean_squared_error as mse
import math
from  imresize import imresize
from random import sample

def check_data(path,data):
    count=0
    for i in range(data.shape[0]):
        if np.max(data[i]) == np.min(data[i]):
            print(i)
            count += 1
    print(f'{path}, {count}, ',end='')

def plt_img(img,idx,name):
    plt.subplot(3, 3, idx)
    plt.imshow(img, cmap='viridis')
    plt.colorbar(shrink=0.6)
    plt.title(f'{name}_img{idx}')
    plt.axis('off')

if __name__=='__main__':
    paths=os.listdir(r'/hdd/zhanghonghu/0001/data/000/train_3000')
    for path in paths:
        hr=np.load(os.path.join(r'/hdd/zhanghonghu/0001/data/000/train_3000',path))
        check_data(path,hr)
        lr=np.load(os.path.join(r'/hdd/zhanghonghu/0001/data/000/train_lr_3000',path))

        fig,axes=plt.subplots(3,3,figsize=(12,12))
        plt.tight_layout()

        sam=sample([i for i in range(len(lr))],3)
        print(sam)
        print('HR_shape:', hr.shape)
        print('LR_shape:', lr.shape)

        for idx,i in enumerate(sam):
            b, h, w=lr.shape

            img = hr[i, :, :]
            img1 = lr[i, :, :]
            cubic_img = imresize(img[:, :, np.newaxis], scale=1 / 4, output_shape=(h, w), kernel='cubic', channel=1)
            cubic_img = cubic_img[:, :, 0]

            plt_img(img, 3*idx+1,'hr')
            plt_img(img1, 3*idx+2,'lr')
            plt_img(cubic_img, 3*idx+3,'cubic_lr')

        plt.show()
