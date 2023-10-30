import argparse
import os
import time

import  torch
from myData import Mydata,Get_data
from    meta import Meta
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #（代表仅使用第0，1号GPU）

# os.makedirs("./images/gan/", exist_ok=True)

torch.manual_seed(111)
# 为CPU设置种子用于生成随机数，以使得结果是确定的。
torch.cuda.manual_seed_all(111)
# 为GPU设置种子用于生成随机数，以使得结果是确定的。
np.random.seed(111)
# 用于生成指定的随机数


def main(args):

    device = torch.device(args.cuda)
    maml = Meta(args).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    data = Mydata()
    get_data = Get_data()
    dataloader = DataLoader(data, batch_size=args.batchsize, shuffle=True)

    glob_step = 0
    writer = SummaryWriter(log_dir='logs')
    val_i=0
    for epoch in range(args.epoches):
        s=time.time()
        l=[]
        for idx, x in enumerate(dataloader):
            T=get_data.in_data(tasks=x,spt=args.spt,qry=args.qry)
            tt0 = maml(T,epoch,idx)
            glob_step = glob_step + 1
            l.append(tt0)
        loss=np.mean(np.array(l),axis=0)
        # 记录训练损失
        writer.add_scalar('Train/Loss', loss[-1], epoch)
        print('epoch:', epoch, ' \ttime:',time.time()-s,'\ttraining loss:', loss)

        if  epoch%2000==0:
            T=get_data.in_fine()
            val_loss=maml.fine(T,epoch)
            # 记录验证损失
            writer.add_scalar('Validation/Loss', val_loss,val_i)
            val_i+=1
    # 关闭SummaryWriter对象
    writer.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoches', type=int, help='epoch number', default=2000000000)
    argparser.add_argument('--cuda', type=str, help='device', default='cuda:0')
    argparser.add_argument('--batchsize', type=int, help='Task num for a single update,  max=len(Tasks)', default=3)
    argparser.add_argument('--update_step', type=int, help='number of updates per task in the inner loop', default=1)

    argparser.add_argument('--spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--qry', type=int, help='k shot for query set', default=10)

    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--task_lr', type=float, help='task-level inner update learning rate', default=1e-2)

    args = argparser.parse_args()
    main(args)