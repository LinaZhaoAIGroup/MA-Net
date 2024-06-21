from .module import *


class UNet(nn.Module):
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
    

class UNet_AG(nn.Module):
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
        
        self.ag1 = AttBlock(64,64)
        self.ag2 = AttBlock(128,128)
        self.ag3 = AttBlock(256,256)
        self.ag4 = AttBlock(512,512)

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


        x = self.de_1_upconv(x)
        en_4 = self.ag4(en_4,x)
        x = self.de_1(torch.concat([en_4,x],dim=1))

        x = self.de_2_upconv(x)
        en_3 = self.ag3(en_3,x)
        x = self.de_2(torch.concat([en_3,x],dim=1))
 
        x = self.de_3_upconv(x)
        en_2 = self.ag2(en_2,x)
        x = self.de_3(torch.concat([en_2,x],dim=1))
 
        x = self.de_4_upconv(x)
        en_1 = self.ag1(en_1,x)
        x = self.de_4(torch.concat([en_1,x],dim=1))
        
        x = self.out_conv(x)
        return x

class UNet_Down_2(nn.Module):
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

        self.en_1 = conv_block(in_ch,64)

        self.df_2 = nn.Conv2d(3,128,3,2,1)
        self.en_2 = conv_block(64+128,128)

        self.df_3 = nn.Conv2d(128,256,3,2,1)
        self.en_3 = conv_block(128+256,256)

        self.df_4 = nn.Conv2d(256,512,3,2,1)
        self.en_4 = conv_block(256+512,512)

        self.df_c = nn.Conv2d(512,1024,3,2,1)
        self.center = conv_block(512+1024,1024)

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
    def forward(self,x):
        en_1 = self.en_1(x)
        x_down = self.df_2(x)
        en_2 = self.en_2(torch.concat([x_down,self.down(en_1)],1))
        x_down = self.df_3(x_down)
        en_3 = self.en_3(torch.concat([x_down,self.down(en_2)],1))
        x_down = self.df_4(x_down)
        en_4 = self.en_4(torch.concat([x_down,self.down(en_3)],1))
        x_down = self.df_c(x_down)
        x = self.center(torch.concat([x_down,self.down(en_4)],1))


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
    
class UNet_DS(nn.Module):
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

        self.en_1 = conv_block(in_ch,64)

        self.ds_2 = Dsampling_Module(64)
        self.en_2 = conv_block(64,128)

        self.ds_3 = Dsampling_Module(128)
        self.en_3 = conv_block(128,256)

        self.ds_4 = Dsampling_Module(256)
        self.en_4 = conv_block(256,512)

        self.ds_c = Dsampling_Module(512)
        self.center = conv_block(512,1024)

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
    def forward(self,x):
        en_1 = self.en_1(x)
        x = self.ds_2(en_1)
        en_2 = self.en_2(x)
        x = self.ds_3(en_2)
        en_3 = self.en_3(x)
        x = self.ds_4(en_3)
        en_4 = self.en_4(x)
        x = self.ds_c(en_4)
        x = self.center(x)


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

class UNet_Up(nn.Module):
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
        self.de_2 = conv_block(256*3,256)
        
        self.de_3_upconv = nn.ConvTranspose2d(256,128,3,2,padding=1,output_padding=1)
        self.de_3 = conv_block(128*4,128)
        
        self.de_4_upconv = nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1)
        self.de_4 = conv_block(64*5,64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64,out_ch,1),
            nn.Sigmoid()
            )
        
        self.up_1_1 = nn.ConvTranspose2d(512,256,3,2,padding=1,output_padding=1)
        self.up_1_2 = nn.ConvTranspose2d(256,128,3,2,padding=1,output_padding=1)
        self.up_1_3 = nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1)
        self.up_2_1 = nn.ConvTranspose2d(256,128,3,2,padding=1,output_padding=1)
        self.up_2_2 = nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1)
        self.up_3_1 = nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1)
        


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


        x = self.de_1_upconv(x)
        u1 = self.up_1_1(x)
        x = self.de_1(torch.concat([en_4,x],dim=1))

        x = self.de_2_upconv(x)
        u2 = self.up_2_1(x)
        x = self.de_2(torch.concat([en_3,x,u1],dim=1))
        u1 = self.up_1_2(u1)
 
        x = self.de_3_upconv(x)
        u3 = self.up_3_1(x)
        x = self.de_3(torch.concat([en_2,x,u1,u2],dim=1))
        u1 = self.up_1_3(u1)
        u2 = self.up_2_2(u2)

        x = self.de_4_upconv(x)
        x = self.de_4(torch.concat([en_1,x,u1,u2,u3],dim=1))
        
        x = self.out_conv(x)
        return x

class UNet_US(nn.Module):
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
        self.de_1_upconv = Usampling_Module(1024,512)
        self.de_1 = conv_block(512*2,512)

        self.de_2_upconv = Usampling_Module(512,256)
        self.de_2 = conv_block(256*2,256)
        
        self.de_3_upconv = Usampling_Module(256,128)
        self.de_3 = conv_block(128*2,128)
        
        self.de_4_upconv = Usampling_Module(128,64)
        self.de_4 = conv_block(64*2,64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64,out_ch,1),
            nn.Sigmoid()
            )
        
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
    

class UNet_CE(nn.Module):
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
        self.de_1_upconv = nn.ConvTranspose2d(1024+4,512,3,2,padding=1,output_padding=1)
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
        self.ce = nn.Sequential(DAC(1024,1024,512), RMP(1024))
        # self.ce = nn.Sequential(DAC(1024,1024+4,512))
        # self.ce = nn.Sequential(conv_block(1024,1024+4))

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
        x = self.ce(x)


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



class MA_UNet(nn.Module):
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

        self.df_2 = nn.Conv2d(3,128,3,2,1)
        self.en_2 = conv_block(64+128,128)

        self.df_3 = nn.Conv2d(128,256,3,2,1)
        self.en_3 = conv_block(128+256,256)

        self.df_4 = nn.Conv2d(256,512,3,2,1)
        self.en_4 = conv_block(256+512,512)

        self.df_c = nn.Conv2d(512,1024,3,2,1)
        self.center = conv_block(512+1024,1024)

        # decoder
        self.de_1_upconv = nn.ConvTranspose2d(1024+4,512,3,2,padding=1,output_padding=1)
        self.de_1 = conv_block(512*2,512)

        self.de_2_upconv = nn.ConvTranspose2d(512,256,3,2,padding=1,output_padding=1)
        self.de_2 = conv_block(256*3,256)
        
        self.de_3_upconv = nn.ConvTranspose2d(256,128,3,2,padding=1,output_padding=1)
        self.de_3 = conv_block(128*4,128)
        
        self.de_4_upconv = nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1)
        self.de_4 = conv_block(64*5,64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64,out_ch,1),
            nn.Sigmoid()
            )
        
        self.up_1_1 = nn.ConvTranspose2d(512,256,3,2,padding=1,output_padding=1)
        self.up_1_2 = nn.ConvTranspose2d(256,128,3,2,padding=1,output_padding=1)
        self.up_1_3 = nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1)
        self.up_2_1 = nn.ConvTranspose2d(256,128,3,2,padding=1,output_padding=1)
        self.up_2_2 = nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1)
        self.up_3_1 = nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1)
        
        # skip_connection
        self.ag1 = AttBlock(64,64)
        self.ag2 = AttBlock(128,128)
        self.ag3 = AttBlock(256,256)
        self.ag4 = AttBlock(512,512)

        # bottleneck
        self.ce = nn.Sequential(DAC(1024,1024,512), RMP(1024))

    def forward(self,x):
        en_1 = self.en_1(x)
        x_down = self.df_2(x)
        en_2 = self.en_2(torch.concat([x_down,self.down(en_1)],1))
        x_down = self.df_3(x_down)
        en_3 = self.en_3(torch.concat([x_down,self.down(en_2)],1))
        x_down = self.df_4(x_down)
        en_4 = self.en_4(torch.concat([x_down,self.down(en_3)],1))
        x_down = self.df_c(x_down)

        x = self.center(torch.concat([x_down,self.down(en_4)],1))
        x = self.ce(x)


        x = self.de_1_upconv(x)
        u1 = self.up_1_1(x)
        en_4 = self.ag4(en_4,x)
        x = self.de_1(torch.concat([en_4,x],dim=1))

        x = self.de_2_upconv(x)
        u2 = self.up_2_1(x)
        en_3 = self.ag3(en_3,x)
        x = self.de_2(torch.concat([en_3,x,u1],dim=1))
        u1 = self.up_1_2(u1)
 
        x = self.de_3_upconv(x)
        u3 = self.up_3_1(x)
        en_2 = self.ag2(en_2,x)
        x = self.de_3(torch.concat([en_2,x,u1,u2],dim=1))
        u1 = self.up_1_3(u1)
        u2 = self.up_2_2(u2)

        x = self.de_4_upconv(x)
        en_1 = self.ag1(en_1,x)
        x = self.de_4(torch.concat([en_1,x,u1,u2,u3],dim=1))
        
        x = self.out_conv(x)
        return x