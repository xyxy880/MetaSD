import numpy as np
import os
import csv

MAX_MIN_PATH='/hdd/zhanghonghu/0001/data/000/train_max_min.csv'

def max_min_normalize(data,max_val,min_val):

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

def lr_normalize():
    paths=os.listdir(r'/hdd/zhanghonghu/0001/data/npy/ori_lr')
    with open(MAX_MIN_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            max_min = {}
            for key,value in row.items():
                max_min[key]=value
            if max_min['name'] in paths:
                filepath=os.path.join(r'/hdd/zhanghonghu/0001/data/npy/ori_lr',max_min['name'])
                data = np.load(filepath)[:, :, :]
                nor = max_min_normalize(data, np.float(max_min['max']), np.float(max_min['min']))
                np.save(os.path.join(r'/hdd/zhanghonghu/0001/data/npy/nor_lr', max_min['name']), nor)

if __name__=='__main__':
    #GFS
    lr_normalize()




















































# import numpy as np
# import os
# import csv
#
# def max_min_normalize(data,max_val,min_val):
#     """
#     对数据进行最大最小归一化
#     :param data: 需要进行归一化的数据，可以是numpy数组或列表
#     :return: 归一化后的数据
#     """
#     result=data[:,:,:]
#     s,h,w=data.shape
#     print(data.shape)
#     for i in range(s):
#         for j in range(h):
#             for k in range(w):
#                 result[i][j][k] = (data[i][j][k] - min_val) / (max_val - min_val)
#     return result
#
# def create_cvs(fieldnames = ['name', 'max', 'min']):
#     with open('max_min.csv', 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#
# def add_data2cvs(new_data,fieldnames = ['name', 'max', 'min']):
#     with open('max_min.csv', 'a', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writerow(new_data)
#
# def lr_guiyihua():
#     paths=os.listdir(r'E:\zhanghonghu\npy_ori\train_lr')
#     with open('max_min.csv', newline='') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             max_min = {}
#             for key,value in row.items():
#                 max_min[key]=value
#             if max_min['name'] in paths:
#                 filepath=os.path.join(r'E:\zhanghonghu\npy_ori\train_lr',max_min['name'])
#                 data = np.load(filepath)[:, :, :]
#                 nor = max_min_normalize(data, np.float(max_min['max']), np.float(max_min['min']))
#                 np.save(os.path.join(r'E:\zhanghonghu\npy_nor\train_lr', max_min['name']), nor)
#                 print(1)
#
#
# if __name__=='__main__':
#     lr_guiyihua()
