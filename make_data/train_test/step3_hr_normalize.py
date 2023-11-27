import numpy as np
import os
import csv
from tqdm import tqdm

MAX_MIN_PATH='/hdd/zhanghonghu/0001/data/000/train_max_min.csv'

def max_min_normalize(data,max_val,min_val):
    """
    对数据进行最大最小归一化
    :param data: 需要进行归一化的数据，可以是numpy数组或列表
    :return: 归一化后的数据
    """
    result=np.divide(data-min_val,max_val-min_val)
    return result

def inverse_max_min_normalize(data,max_value,min_value):

    result= np.multiply(data,(max_value - min_value)) + min_value
    return result

def create_cvs(fieldnames = ['name', 'max', 'min']):
    with open(MAX_MIN_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def add_data2cvs(new_data,fieldnames = ['name', 'max', 'min']):
    with open(MAX_MIN_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(new_data)
if __name__=='__main__':

    paths=os.listdir(r'/hdd/zhanghonghu/0001/data/npy/ori')
    if not os.path.exists(MAX_MIN_PATH):
        create_cvs()

    for path in tqdm(paths):
        filepath=os.path.join(r'/hdd/zhanghonghu/0001/data/npy/ori',path)
        data=np.load(filepath)[:,:,:]
        max,min=np.max(data),np.min(data)
        add_data2cvs({'name':path,
                      'max':max,
                      'min':min})
        if not os.path.exists(os.path.join(r'/hdd/zhanghonghu/0001/data/npy/nor',path)):
            nor=max_min_normalize(data,max,min)
            np.save(os.path.join(r'/hdd/zhanghonghu/0001/data/npy/nor',path),nor)
