import os
import numpy as np

if __name__ == '__main__':
    paths=os.listdir('/hdd/zhanghonghu/0001/data/npy/nor')
    for path in paths:
        data=np.load(os.path.join('/hdd/zhanghonghu/0001/data/npy/nor',path))
        np.save(os.path.join('/hdd/zhanghonghu/0001/data/000/train_3000',path),data[:3000,:,:])
        np.save(os.path.join('/hdd/zhanghonghu/0001/data/000/test_100',path),data[3000:3100,:,:])

    paths=os.listdir('/hdd/zhanghonghu/0001/data/npy/nor_lr')
    for path in paths:
        data=np.load(os.path.join('/hdd/zhanghonghu/0001/data/npy/nor_lr',path))
        np.save(os.path.join('/hdd/zhanghonghu/0001/data/000/train_lr_3000',path),data[:3000,:,:])
        np.save(os.path.join('/hdd/zhanghonghu/0001/data/000/test_lr_100',path),data[3000:3100,:,:])
