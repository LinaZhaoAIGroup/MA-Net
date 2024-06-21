import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import *


class Patch_Embedding(nn.Module):
    def __init__(self,patch_size=8):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size,patch_size)
    def forward(self,x):
        x = self.pool(x)
        x = einops.rearrange(x,"b c h w -> b (h w) c")
        return x

class Projection(nn.Module):
    def __init__(self,in_ch,out_ch) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self,x):
        H = int(x.shape[1]**0.5)
        x = einops.rearrange(x, "B (H W) C -> B C H W",H=H)
        x = self.conv(x)
        x = einops.rearrange(x, "B C H W -> B (H W) C",H=H)
        return x


class CCA(nn.Module):
    def __init__(self,features) -> None:
        super().__init__()
        f_sum = np.sum(features)
        self.norm_list = nn.ModuleList([
            nn.LayerNorm(f) for f in features
            ])
        self.q_map_list = nn.ModuleList([
            Projection(f,f) for f in features
        ])
        self.k_map = Projection(f_sum,f_sum)
        self.v_map = Projection(f_sum,f_sum)

        self.proj_list = nn.ModuleList([
            Projection(f,f) for f in features
        ])
    def forward(self,x_list):
        # B L C
        x_list = [n(x) for n,x in zip(self.norm_list,x_list)]
        # B L C
        Q_list = [m(x) for m,x in zip(self.q_map_list,x_list)]
        # B L C_sum
        K = self.k_map(torch.concat(x_list,dim=-1))
        # B L C_sum
        V = self.v_map(torch.concat(x_list,dim=-1))
        scale = V.shape[-1]**0.5
        x_list = [F.softmax(Q.transpose(1,2)@K/scale,dim=-1)@V.transpose(1,2) for Q in Q_list]
        x_list = [p(x.transpose(1,2)) for p,x in zip(self.proj_list,x_list)]
        return x_list

class SCA(nn.Module):
    def __init__(self,features,head_num=4) -> None:
        super().__init__()
        self.norm_list = nn.ModuleList([
            nn.LayerNorm(f) for f in features
            ])
        f_sum = np.sum(features)
        self.q_map = Projection(f_sum,f_sum)
        self.k_map = Projection(f_sum,f_sum)
        self.v_map_list = nn.ModuleList([
            Projection(f,f) for f in features
        ])
        self.proj_list = nn.ModuleList([
            Projection(f,f) for f in features
        ])

        self.head_num=head_num

    def forward(self,x_list):
        # B L C
        x_list = [n(x) for n,x in zip(self.norm_list,x_list)]
        # B L C_sum
        Q = self.q_map(torch.concat(x_list,dim=-1))
        # B L C_sum
        K = self.k_map(torch.concat(x_list,dim=-1))
        # B L C
        V_list = [m(x) for m,x in zip(self.v_map_list,x_list)]
        Q = einops.rearrange(Q,"B L (H C) -> B H L C",H=self.head_num)
        K = einops.rearrange(K,"B L (H C) -> B H L C",H=self.head_num)
        V_list = [einops.rearrange(x,"B L (H C) -> B H L C",H=self.head_num) for x in V_list]
        

        scale = Q.shape[-1]**0.5
        x_list = [F.softmax(Q@K.transpose(-1,-2)/scale,dim=-1)@V for V in V_list]
        x_list = [einops.rearrange(x,"B H L C -> B L (H C)",H=self.head_num) for x in x_list]
        x_list = [p(x) for p,x in zip(self.proj_list,x_list)]
        return x_list


class DCA(nn.Module):
    def __init__(self,features,patch_size) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pool_list = nn.ModuleList([Patch_Embedding(p) for p in patch_size])
        self.proj_list = nn.ModuleList([Projection(f,f) for f in features])
        self.cca = CCA(features)
        self.sca = SCA(features)
        self.norm_list = nn.ModuleList([nn.LayerNorm(f) for f in features])
        self.act = nn.GELU()

        self.conv_list = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(f,f,3,1,1),
            nn.BatchNorm2d(f),
            nn.ReLU()
        )for f in features])

    def up(self,x_list):
        x_list = [F.interpolate(x, size=(x.shape[-1]*s,x.shape[-1]*s), mode="bilinear") for x,s in zip(x_list,self.patch_size)]
        return x_list

    def forward(self,x_list):
        x_list = [p(x) for p,x in zip(self.pool_list,x_list)]
        x_list = [p(x) for p,x in zip(self.proj_list,x_list)]
        cca_list = self.cca(x_list)
        x_list = [x+c for c,x in zip(cca_list,x_list)]
        sca_list = self.sca(x_list)
        x_list = [x+s for s,x in zip(sca_list,x_list)]
        x_list = [einops.rearrange(x,"B (H W) C -> B C H W",H=int(x.shape[1]**0.5)) for x in x_list]
        x_list = self.up(x_list)
        x_list = [c(x) for c,x in zip(self.conv_list,x_list)]

        return x_list



class DCA_UNet(nn.Module):
    def up(self,x):
        _,_,h,w = x.shape
        return F.interpolate(x, size=(int(h*2),int(w*2)), mode="bilinear")
    def down(self, x):
        return F.max_pool2d(x, kernel_size=2)
    def __init__(self, in_ch=3,
                  out_ch=1,
                  use_res_block=True # 是否使用残差卷积块
                    ) -> None:
        super().__init__()
        conv_block=Res_Conv_Block if use_res_block else Conv_Block

        # encoder
        self.en_1 = conv_block(in_ch,64)
        self.en_2 = conv_block(64,128)
        self.en_3 = conv_block(128,256)
        self.en_4 = conv_block(256,512)
        self.center = conv_block(512,1024)

        # decoder
        self.de_1_upconv = nn.ConvTranspose2d(1024,512,3,2,padding=1,output_padding=1)
        self.de_1 = conv_block(512*2,512)

        self.de_2_upconv = nn.ConvTranspose2d(512,256,3,2,padding=1,output_padding=1)
        self.de_2 = conv_block(256*2,256)
        
        self.de_3_upconv = nn.ConvTranspose2d(256,128,3,2,padding=1,output_padding=1)
        self.de_3 = conv_block(128*2,128)
        
        self.de_4_upconv = nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1)
        self.de_4 = conv_block(64*2,64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64,out_ch,1),
            nn.Sigmoid()
            )
        
        self.dca = DCA([64,128,256,512],[16,8,4,2])
    def forward(self,x):
        en_1 = self.en_1(x)
        x = self.down(en_1)

        en_2 = self.en_2(x)
        x = self.down(en_2)

        en_3 = self.en_3(x)
        x = self.down(en_3)

        en_4 = self.en_4(x)
        x = self.down(en_4)

        x = self.center(x)

        [en_1,en_2,en_3,en_4] = self.dca([en_1,en_2,en_3,en_4])

        x = self.de_1_upconv(x)
        x = self.de_1(torch.concat([en_4,x],dim=1))

        x = self.de_2_upconv(x)
        x = self.de_2(torch.concat([en_3,x],dim=1))
 
        x = self.de_3_upconv(x)
        x = self.de_3(torch.concat([en_2,x],dim=1))
 
        x = self.de_4_upconv(x)
        x = self.de_4(torch.concat([en_1,x],dim=1))
        
        x = self.out_conv(x)
        return x
   