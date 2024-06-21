from .maxvit import *
from timm.models._efficientnet_blocks import ConvBnAct

class MaxVit_Unet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stem = nn.Sequential(
            ConvBnAct(3,64,3,2,1),
            ConvBnAct(64,64,3,1,1),
            )
        self.s1 = nn.Sequential(
            MaxViTBlock(64,64,downscale=True,num_heads=2,grid_window_size=(8,8)),
            MaxViTBlock(64,64,num_heads=2,grid_window_size=(8,8))
        )
        self.s2 = nn.Sequential(
            MaxViTBlock(64,128,downscale=True,num_heads=2,grid_window_size=(8,8)),
            MaxViTBlock(128,128,num_heads=2,grid_window_size=(8,8))
        )
        self.s3 = nn.Sequential(
            MaxViTBlock(128,256,downscale=True,num_heads=2,grid_window_size=(8,8)),
            MaxViTBlock(256,256,num_heads=2,grid_window_size=(8,8))
        )
        self.s4 = nn.Sequential(
            MaxViTBlock(256,512,downscale=True,num_heads=2,grid_window_size=(8,8)),
            MaxViTBlock(512,512,num_heads=2,grid_window_size=(8,8))
        )

        self.up3 = nn.ConvTranspose2d(512,256,2,2)
        self.d3 = nn.Sequential(
            MaxViTBlock(512,256,num_heads=2,grid_window_size=(8,8)),
            MaxViTBlock(256,256,num_heads=2,grid_window_size=(8,8))
        )

        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.d2 = nn.Sequential(
            MaxViTBlock(256,128,num_heads=2,grid_window_size=(8,8)),
            MaxViTBlock(128,128,num_heads=2,grid_window_size=(8,8))
        )

        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.d1 = nn.Sequential(
            MaxViTBlock(128,64,num_heads=2,grid_window_size=(8,8)),
            MaxViTBlock(64,64,num_heads=2,grid_window_size=(8,8))
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64,64,2,2),
            ConvBnAct(64,64,3,1,1),
            nn.ConvTranspose2d(64,64,2,2),
            nn.Conv2d(64,1,3,1,1),
            nn.Sigmoid()
        )


    def forward(self,x):
        x = self.stem(x)
        
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)

        x = torch.concat([s3,self.up3(s4)],1)
        x = self.d3(x)
        x = torch.concat([s2,self.up2(s3)],1)
        x = self.d2(x)
        x = torch.concat([s1,self.up1(s2)],1)
        x = self.d1(x)

        x = self.out(x)
        return x
        