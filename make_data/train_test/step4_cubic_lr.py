import os
import numpy as np
from  imresize import imresize

def cubic(data):
    s,h,w=data.shape
    result=[]
    for i in range(s):
        img=data[i][:,:,np.newaxis]
        lr = imresize(img, scale=1 / 4, output_shape=(h//4, w//4), kernel='cubic',channel=1)
        lr = lr[:,:,0]
        result.append(lr)
    result=np.array(result)
    return result

if __name__=='__main__':
    #ERA5
    paths=os.listdir(r'/hdd/zhanghonghu/0001/data/npy/nor')
    paths = [path for path in paths if "GFS" not in path]
    for path in paths:
        filepath=os.path.join(r'/hdd/zhanghonghu/0001/data/npy/nor',path)
        filename=os.path.join(r'/hdd/zhanghonghu/0001/data/npy/nor_lr',path)
        if os.path.exists(filename):
            continue
        data=np.load(filepath)
        result=cubic(data)
        np.save(filename,result)
