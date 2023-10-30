import  torch
from    torch import nn
from    torch.nn import functional as F


class EDSR4(nn.Module):

    def __init__(self):

        super(EDSR4, self).__init__()

        self.B=16 #Number of residual blocks
        self.scale=4 #upscaling factor

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()

        #0-63
        for i in range(0,self.B*2):
            w = nn.Parameter(torch.nn.init.kaiming_normal_(torch.ones(64, 64, 3, 3)))
            b = nn.Parameter(torch.zeros(64))
            self.vars.extend([w, b])

        self.vars.extend([
            nn.Parameter(torch.nn.init.kaiming_normal_(torch.ones(64, 1, 3, 3))),  # 64
            nn.Parameter(torch.zeros(64)),  # 65

            nn.Parameter(torch.nn.init.kaiming_normal_(torch.ones(64, 64, 3, 3))),  # 66
            nn.Parameter(torch.zeros(64)),  # 67

            nn.Parameter(torch.nn.init.kaiming_normal_(torch.ones(16, 64, 3, 3))),  # 68
            nn.Parameter(torch.zeros(16))  # 69
        ])

    def forward(self, x,vars=None):
        """
        :param x: [b, 1, h, w]
        :param vars:
        :return :
        """

        if vars is None:
            vars = self.vars

        #64-65
        idx = 64
        w, b = vars[idx], vars[idx + 1]
        x=F.conv2d(x,w,b,stride=1,padding=(w.shape[-1]-1)//2)
        x=F.relu(x)
        out1=x

        for idx in range(0,self.B*4,4):
            t=x
            w, b = vars[idx], vars[idx + 1]
            x = F.conv2d(x, w, b, stride=1, padding=(w.shape[-1] - 1) // 2)
            x = F.relu(x)

            w, b = vars[idx+2], vars[idx + 3]
            x = F.conv2d(x, w, b, stride=1, padding=(w.shape[-1] - 1) // 2)

            x=t+x*0.1

        #66-67
        idx = 66
        w, b = vars[idx], vars[idx + 1]
        x = F.conv2d(x, w, b, stride=1, padding=(w.shape[-1] - 1) // 2)
        x = F.relu(x)
        x=x+out1


        #68-69
        idx=68
        w, b = vars[idx], vars[idx + 1]
        x=F.conv2d(x,w,b,stride=1,padding=(w.shape[-1]-1)//2)
        x=torch.pixel_shuffle(x, 4)

        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

if __name__=='__main__':
    x=torch.ones(10,1,40,12)
    net=EDSR4()
    total_params = sum(p.numel() for p in net.parameters())
    y=net(x,None)

    print("Total number of parameters: ", total_params)
    print(y.shape)
    print(net)



