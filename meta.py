from torch import optim
import  numpy as np
from myLoss import *
from model.edsr_x4 import EDSR4
from utils import task_weight,get_start_lr
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import cv2 as cv
from copy import deepcopy

class Meta(nn.Module):

    def __init__(self,args):

        super(Meta, self).__init__()
        self.device = args.cuda
        self.task_lr = args.task_lr
        self.meta_lr = args.meta_lr
        self.net = EDSR4().cuda()

        # 加载预训练模型参数
        # self.net.load_state_dict(torch.load('/hdd/zhanghonghu/0001/001/code/ZHH_MAML/version6_0/15001.pth'))
        # self.loss_fn=My_loss()

        self.loss_fn=torch.nn.L1Loss().cuda()
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.update_step=args.update_step
        self.batchsize=args.batchsize

    def forward(self,Tasks,epoch,idx):
        #save model
        if epoch%2000==0 and idx==0:
            torch.save(self.net.state_dict(),'./pth/'+str(epoch)+'.pth')

        loss_qq = [0 for _ in range(self.update_step + 1)]
        q=[]
        for task in Tasks:
            cur_loss=[]
            x_spt, y_spt, x_qry, y_qry = task[0], task[1], task[2], task[3]
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt.astype(np.float32)).to(self.device), torch.from_numpy(y_spt.astype(np.float32)).to(self.device), torch.from_numpy(x_qry.astype(np.float32)).to(self.device), torch.from_numpy(y_qry.astype(np.float32)).to(self.device)

            # 0 updates
            with torch.no_grad():
                pred0_qry = self.net(x_qry, self.net.parameters())
                loss0 = self.loss_fn(pred0_qry, y_qry)
                loss_qq[0] += loss0.item()
                cur_loss.append(loss0)

            #1 update
            pred1_spt = self.net(x_spt, vars=None)
            loss1 = self.loss_fn(pred1_spt, y_spt)
            grad1 = torch.autograd.grad(loss1, self.net.parameters(),create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad1, self.net.parameters())))
            #query set
            pred1_qry = self.net(x_qry, fast_weights)
            loss1 = self.loss_fn(pred1_qry, y_qry)
            loss_qq[1] += loss1.item()
            cur_loss.append(loss1)

            #2~n update
            for i in range(1,self.update_step):
                predi_spt = self.net(x_spt, vars=fast_weights)
                lossi = self.loss_fn(predi_spt, y_spt)
                gradi = torch.autograd.grad(lossi, fast_weights, create_graph=True)
                fast_weights=list(map(lambda p: p[1] - self.task_lr * p[0], zip(gradi, fast_weights)))
                #query set
                predi_qry = self.net(x_qry, fast_weights)
                lossi = self.loss_fn(predi_qry, y_qry)
                loss_qq[i+1] += lossi.item()
                cur_loss.append(lossi)
            q.append(cur_loss)

        #Weighted loss
        li=[i[-1] for i in q]
        weight=task_weight(li)
        loss=sum([loss*w for loss,w in zip(li,weight)])

        #meta update
        self.meta_optim.zero_grad()
        loss.backward()
        self.meta_optim.step()



        return [t/self.batchsize for t in loss_qq]

    def mean_psr(self,imgs1, imgs2, data_range):
        # imgs1_shape:B*H*W
        PP = []
        SS = []
        RR = []
        for img1, img2 in zip(imgs1, imgs2):
            P = psnr(img1, img2, data_range=data_range)
            S = ssim(img1, img2, data_range=data_range)
            R = np.sqrt(mse(img1, img2))
            PP.append(P)
            SS.append(S)
            RR.append(R)
        return np.mean(PP), np.mean(SS), np.mean(RR)

    def fine(self,Tasks,epoch):
        net = deepcopy(self.net)
        net=net.to(self.device)
        # loss_fn = My_loss().to(self.device)
        opt=optim.SGD(net.parameters(), lr=self.task_lr)

        val_loss=[]
        for task in Tasks:
            x_fine, y_fine, x_test, y_test = task[0], task[1], task[2], task[3]
            x_fine, y_fine, x_test, y_test = torch.from_numpy(x_fine.astype(np.float32)).to(self.device), torch.from_numpy(
                y_fine.astype(np.float32)).to(self.device), torch.from_numpy(x_test.astype(np.float32)).to(
                self.device), torch.from_numpy(y_test.astype(np.float32)).to(self.device)

            for i in range(30):
                if i >= 1:
                    if i == 1:
                        lr = get_start_lr(x_fine[0,0,:,:].cpu().detach().numpy(), y_fine[0,0,:,:].cpu().detach().numpy())
                        opt = optim.Adam(net.parameters(), lr=lr)
                    pred = net(x_fine)
                    loss = self.loss_fn(pred, y_fine)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    continue
                pred = net(x_fine)
                loss = self.loss_fn(pred, y_fine)
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                pred=net(x_test)
                loss=self.loss_fn(pred,y_test)
                val_loss.append(loss.item())

                # SR = np.clip(pred[:, 0, :, :].cpu().detach().numpy(),0.,1.)
                # HR = y_test[:, 0, :, :].cpu().detach().numpy()
                # LR =x_test[:, 0, :, :].cpu().detach().numpy()
        del net

        return np.mean(np.array(val_loss))


if __name__ == '__main__':
    pass
