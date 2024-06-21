import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv

class Basic_Conv(nn.Module):
    def __init__(self, in_ch,out_ch,ksize,stride=1,padding=0,dilation=1,groups=1,) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,ksize,stride,padding,dilation,groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)

class Conv_Block(nn.Module):
    def __init__(self, in_dim,out_dim,
                 use_bn=True
                 ) -> None:
        super().__init__()
        dim_list = [in_dim,in_dim,out_dim]
        conv_list = []
        for i in range(len(dim_list)-1):
            dim1,dim2 = dim_list[i:i+2]
            conv_list.append(nn.Conv2d(dim1,dim2,3,1,1))
            if use_bn:
                conv_list.append(nn.BatchNorm2d(dim2))
            conv_list.append(nn.ReLU())
        self.conv = nn.ModuleList(conv_list)
    def forward(self,x):
        for c in self.conv:
            x = c(x)
        return x

class Res_Conv_Block(nn.Module):
    def __init__(self, in_dim,out_dim,
                 use_bn=True
                 ) -> None:
        super().__init__()
        dim_list = [in_dim,in_dim,out_dim]
        conv_list = []
        for i in range(len(dim_list)-1):
            dim1,dim2 = dim_list[i:i+2]
            conv_list.append(nn.Conv2d(dim1,dim2,3,1,1))
            if use_bn:
                conv_list.append(nn.BatchNorm2d(dim2))
            conv_list.append(nn.ReLU())
        self.conv = nn.ModuleList(conv_list)
        self.res_conv = nn.Conv2d(in_dim,out_dim,1)
    def forward(self,x):
        x_res = self.res_conv(x)
        for c in self.conv:
            x = c(x)
        return x+x_res

# 附加模块
# 深度可分离卷积
class DepthWise_Conv(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 k_size = 3,
                 stride = 1,
                 padding = 1,
                 use_res=True # 是否使用残差连接
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,in_ch,k_size,stride,padding,groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch,out_ch,1,1)
        if use_res:
            self.res = nn.Conv2d(in_ch,out_ch,1)
        else:
            self.res = None
    def forward(self,x):
        if self.res is not None:
            return self.res(x) + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))
# AG
class AttBlock(nn.Module):
    def __init__(self, x_dim,g_dim) -> None:
        super().__init__()
        self.wx = nn.Conv2d(x_dim,1,1)
        self.wg = nn.Conv2d(g_dim,1,1)
        self.phi = nn.Conv2d(1,1,1)
    def forward(self,x,g):
        att = F.relu(self.wx(x)+self.wg(g))
        att = F.sigmoid(self.phi(att))
        return x*att

# DAC模块
class DAC(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 hidden_dim=None
                  ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim,hidden_dim,1,1,0),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
            )
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim,out_dim,1,1,0),
            nn.ReLU()
            )
        
        self.b_1 = nn.Conv2d(hidden_dim,hidden_dim,3,1,1)
        self.b_2 = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,3,1,3,dilation=3),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0),
            nn.ReLU()
            )
        self.b_3 = nn.Identity()
        self.b_4 = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,3,1,1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim,3,1,3,dilation=3),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0),
            nn.ReLU()
            )
        self.b_5 = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,3,1,1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim,3,1,3,dilation=3),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim,3,1,5,dilation=5),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0),
            nn.ReLU()
            )
        
    def forward(self,x):
        x = self.in_conv(x)
        x_1,x_2,x_3,x_4,x_5 = self.b_1(x), self.b_2(x), self.b_3(x), self.b_4(x), self.b_5(x)
        x = x_1+x_2+x_3+x_4+x_5
        x = self.out_conv(x)
        return x


class RMP(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.p1 = nn.MaxPool2d(2)
        self.p2 = nn.MaxPool2d(3)
        self.p3 = nn.MaxPool2d(5)
        self.p4 = nn.MaxPool2d(6)
        self.conv = nn.Conv2d(in_ch,1,1,1,0)
    def up(self,x,size):
        return F.interpolate(x, size, mode="bilinear")
    def forward(self,x):
        _,_,h,w = x.shape
        size=(h,w)
        x1, x2, x3, x4 = self.conv(self.p1(x)), self.conv(self.p2(x)), self.conv(self.p3(x)), self.conv(self.p4(x))
        x1, x2, x3, x4 = self.up(x1,size),self.up(x2,size),self.up(x3,size),self.up(x4,size)
        return torch.concat([x1, x2, x3, x4, x],1)


# Laplacian Pyramid
# class LP(nn.Module):
#     def __init__(self,layer_num,
#                  in_ch=3,
#                  k_size=3,
#                  sigma=1.0
#                  ):
#         super().__init__()
#         self.layer_num = layer_num
#         k = cv.getGaussianKernel(k_size,sigma,cv.CV_32F)
#         k = torch.tensor(k@k.T)
#         k = torch.stack([k for _ in range(in_ch)]).unsqueeze(1)

#         self.conv = nn.Conv2d(in_ch, in_ch, k_size, bias=False,groups=in_ch,padding=int(k_size/2),padding_mode="reflect")
#         self.conv.weight.data = k
#         self.conv.weight.requires_grad = False

#     def down(self,x):
#         return F.max_pool2d(x, kernel_size=2)
#     def up(self,x):
#         _,_,h,w = x.shape
#         return F.interpolate(x, size=(int(h*2),int(w*2)), mode="nearest")
#     def forward(self,x):
#         # 高斯金字塔
#         g_ps = []
#         for _ in range(self.layer_num+1):
#             x = self.conv(x)
#             g_ps.append(x)
#             x = self.down(x)
#         # 拉普拉斯金字塔
#         l_ps = []
#         for i in range(self.layer_num):
#             l_ps.append(g_ps[i]-self.up(g_ps[i+1]))

#         return g_ps[:self.layer_num],l_ps[:self.layer_num]


# 多尺度下采样模块
class Dsampling_Module(nn.Module):
    def __init__(self,in_ch) -> None:
        super().__init__()
        out_ch = int(in_ch/4)
        self.pool_1 = nn.Sequential(
            nn.MaxPool2d(2),
            Basic_Conv(in_ch,out_ch,1)
            )
        self.pool_2 = nn.Sequential(
            nn.AvgPool2d(2),
            Basic_Conv(in_ch,out_ch,1)
            )
        self.pool_3 = Basic_Conv(in_ch,out_ch,3,2,1)
        self.pool_4 = Basic_Conv(in_ch,out_ch,5,2,2)
        self.res_pool = nn.MaxPool2d(2,2)
    def forward(self,x):
        x = self.res_pool(x)+torch.concat([self.pool_1(x),self.pool_2(x),self.pool_3(x),self.pool_4(x)],1)
        return x

# 上采样模块
class Usampling_Module(nn.Module):
    def __init__(self, in_ch,out_ch) -> None:
        super().__init__()
        out_ch = int(out_ch/2)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,3,2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()            
            )
        self.conv2 = Basic_Conv(in_ch,out_ch,1)

    def forward(self,x):
        _,_,h,w = x.shape
        x = torch.concat([self.conv1(x),self.conv2(F.interpolate(x, size=(int(h*2),int(w*2)), mode="bilinear"))],dim=1)
        return x