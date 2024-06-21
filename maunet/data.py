import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class SegDataset(Dataset):
    def __init__(self,
                 path,
                 resize=512,
                 data_select=None
                 ):
        super().__init__()
        img_root = os.path.join(path,"img")
        mask_root = os.path.join(path,"mask")
        self.img = np.array([os.path.join(img_root,f)for f in os.listdir(img_root)])
        self.mask = np.array([os.path.join(mask_root,f)for f in os.listdir(mask_root)])
        self.img.sort()
        self.mask.sort()

        if data_select is not None:
            self.img = self.img[data_select>0]
            self.mask = self.mask[data_select>0]

        self.transform = transforms.Compose([
            transforms.Resize(resize)
            ])
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, index):
        img = Image.open(self.img[index]).convert("RGB")
        mask = Image.open(self.mask[index])
        img = np.array(self.transform(img))
        mask = np.array(self.transform(mask))

        img = img/255
        mask[mask>0]=1

        img = torch.tensor(img).permute(2,0,1).float()
        mask = torch.tensor(mask).long()

        return img,mask

# k折交叉验证
def get_k_fold(path,
               i, # 取第i折作为测试集
               k=5
               ):
    data_num = len(SegDataset(path))
    val_num = int(data_num/k)
    train_select = np.ones(data_num)
    train_select[int(i*val_num):int((i+1)*val_num)]=0
    val_select = np.ones_like(train_select)-train_select

    train_set = SegDataset(path,data_select=train_select)
    val_set = SegDataset(path,data_select=val_select)

    return train_set,val_set