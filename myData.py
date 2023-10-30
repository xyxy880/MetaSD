from torch.utils.data import Dataset,DataLoader
import time
import numpy as np
import os
from random import sample

class Mydata(Dataset):

    def __init__(self):
        super(Mydata, self).__init__()
        self.data=[i for i in range(18)]

    def __getitem__(self, index):
        x=self.data[index]
        return x

    def __len__(self):
        return len(self.data)

class Get_data():
    def __init__(self):
        super(Get_data, self).__init__()
        self.Tasks_hr=[]
        self.Tasks_lr=[]
        self.Tasks_test_hr=[]
        self.Tasks_test_lr=[]

        self.l=[i for i in range(3000)]

        paths=os.listdir('/hdd/zhanghonghu/0001/001/nc/new/train_3000')
        with open('idx_task.txt', 'w') as f:
            for idx,path in enumerate(paths):
                f.write(f'{idx}\t{path}\n')

        s=time.time()
        for path in paths:
            self.Tasks_hr.append(np.load(os.path.join('/hdd/zhanghonghu/0001/001/nc/new/train_3000', path)))
            self.Tasks_lr.append(np.load(os.path.join('/hdd/zhanghonghu/0001/001/nc/new/train_lr_3000', path)))

            self.Tasks_test_hr.append(np.load(os.path.join('/hdd/zhanghonghu/0001/001/nc/new/test_100', path)))
            self.Tasks_test_lr.append(np.load(os.path.join('/hdd/zhanghonghu/0001/001/nc/new/test_lr_100', path)))

        print(time.time()-s)
    def in_data(self,tasks,spt=3,qry=3):

        T=[]
        for task in tasks:
            t=[]
            sam = sample(self.l, spt + qry)

            x_spts = self.Tasks_lr[task][sam[:spt],np.newaxis, :, :]
            y_spts = self.Tasks_hr[task][sam[:spt],np.newaxis, :, :]
            x_qrys = self.Tasks_lr[task][sam[spt:],np.newaxis, :, :]
            y_qrys = self.Tasks_hr[task][sam[spt:], np.newaxis,:, :]

            t.append(x_spts)
            t.append(y_spts)
            t.append(x_qrys)
            t.append(y_qrys)

            T.append(t)
        return T
    def in_fine(self):
        T=[]
        for task in range(18):
            t=[]
            sam = [i for i in range(100)]

            x_fine = self.Tasks_test_lr[task][sam[:2],np.newaxis, :, :]
            y_fine = self.Tasks_test_hr[task][sam[:2],np.newaxis, :, :]
            x_test = self.Tasks_test_lr[task][sam[2:4],np.newaxis, :, :]
            y_test = self.Tasks_test_hr[task][sam[2:4], np.newaxis,:, :]

            t.append(x_fine)
            t.append(y_fine)
            t.append(x_test)
            t.append(y_test)

            T.append(t)
        return T

if __name__=='__main__':

    data=Mydata()
    get_data= Get_data()
    dataloader=DataLoader(data,batch_size=2,shuffle=False)
    for idx,x in enumerate(dataloader):
        T=get_data.in_data(tasks=x)
        print(1)
        # time.sleep(2)



