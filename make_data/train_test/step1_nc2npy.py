import os
import numpy as np
from netCDF4 import Dataset
import numpy as np
from random import sample
'''
data
----nc
----grib
--------GFS
----npy
--------ori
--------ori_lr
--------nor
--------nor_lr
----000
--------train_3000
--------train_lr_3000
--------test_100
--------test_lr_100
'''

def check_data(x,lis):

    for li in lis:
        count = 0
        factor = x.variables[li][:, :, :]
        for i in range(factor.shape[0]):

            if np.max(factor[i]) == np.min(factor[i]):
                print(i)
                count += 1
        print(li,count)

if __name__=='__main__':

    # xx = Dataset(r'/hdd/zhanghonghu/0001/data/nc/2000.nc')
    # lis = ['t2m','d2m','u10','v10','sp','tp']
    # check_data(xx, lis)
    # for li in lis:
    #     factor = xx.variables[li][:, :, :]
    #     sam=sample([i for i in range(len(factor))],3100)
    #     factor_tr=np.array(factor[sam,:,:])
    #     np.save(fr'/hdd/zhanghonghu/0001/data/npy/ori/era5_2000_{li}.npy',factor_tr)

    xx = Dataset(r'/hdd/zhanghonghu/0001/data/nc/2020.nc')
    lis = ['t2m','d2m','u10','v10','sp','tp']
    check_data(xx, lis)
    for li in lis:
        factor = xx.variables[li][:, :, :]
        sam=sample([i for i in range(len(factor))],3100)
        factor_tr=np.array(factor[sam,:,:])
        np.save(fr'/hdd/zhanghonghu/0001/data/npy/ori/era5_2020_{li}.npy',factor_tr)





