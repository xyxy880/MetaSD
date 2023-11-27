import numpy as np
import pygrib
import os
from random import sample

def get_npyData(li,grib_dir,save_dir,sam=None):
    variable = {
         't2m': '2 metre temperature',
         'd2m': '2 metre dewpoint temperature',
         'u10': '10 metre U wind component',
         'v10': '10 metre V wind component',
         'sp': 'Surface pressure',
         'tp': 'Total Precipitation',
         }
    result=[]

    for i in ['0731','0801','0802','0803','0804','0805','0806']:
        for j in ['00','06','12','18']:
            for k in range(3,387,3):
                filepath=os.path.join(grib_dir,li,'2023'+i,f'{j}_{k}.grib')
                grbs = pygrib.open(filepath)
                grbs.seek(0)
                # for grb in grbs:
                #     print(grb)
                grb = grbs.select(name=variable[li])[0]
                value = grb.values[:,:]
                result.append(value)

    result=np.array(result)
    if sam==None:
        sam=sample([i for i in range(len(result))],3100)
    np.save(fr'{save_dir}/GFS_2023_{li}.npy', result[sam,:,:])
    return sam

if __name__=='__main__':

    lis=['t2m','d2m','u10','v10','sp','tp']
    for li in lis:
        #0.25°
        sam=get_npyData(li,grib_dir='/hdd/zhanghonghu/0001/data/grib/GFS/025',save_dir='/hdd/zhanghonghu/0001/data/npy/ori',sam=None)
        #1°
        get_npyData(li,grib_dir='/hdd/zhanghonghu/0001/data/grib/GFS/100',save_dir='/hdd/zhanghonghu/0001/data/npy/ori_lr',sam=sam)


