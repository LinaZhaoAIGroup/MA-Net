import torch
import os
import logging
from tqdm import tqdm
import numpy as np
from .utils import *

class Trainer:
    def __init__(self,model,optim,loss_func,save_path,device,scheduler=None):
        self.model = model.to(device)

        self.optim = optim

        self.loss_func = loss_func
         
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
        self.device = device

        self.scheduler = scheduler

        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers=[]
        fh = logging.FileHandler(os.path.join(self.save_path,'log.log'),"w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
       
        # 模型参数量
        self.logger.info(f"model_param_count:\t{model_param_count(self.model)/1024/1024:.3f} M")

    def train_one_epoch(self,train_loader):
        self.model.train()
        loss_list = []
        for imgs,labe in tqdm(train_loader):
            imgs = imgs.to(self.device)
            labe = labe.to(self.device)

            pred = self.model(imgs)
            loss = self.loss_func(pred,labe)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            loss_list.append(loss.item())
        return np.mean(loss_list)
    
    def val_one_epoch(self,val_loader):
        self.model.eval()
        loss_list = []
        for imgs,label in tqdm(val_loader):
            imgs = imgs.to(self.device)
            label = label.to(self.device)

            pred = self.model(imgs)
            loss = self.loss_func(pred,label)
            loss_list.append(loss.item())
        return np.mean(loss_list)

    def train(self,train_loader,val_loader,epoches,max_e=None):
        loss_min = np.inf
        
        if max_e is None:
            max_e = epoches

        e_count = max_e
        for e in range(1,epoches+1):
            info = f"{e}/{epoches}\t{e_count}/{max_e}"

            train_loss = self.train_one_epoch(train_loader)
            info += f"\ttrain_loss:{train_loss}"

            val_loss = self.val_one_epoch(val_loader)
            info += f"\tval_loss:{val_loss}"
            # torch.cuda.empty_cache()
            
            if val_loss<loss_min:
                loss_min=val_loss
                torch.save(self.model,os.path.join(self.save_path,"model.pth"))
                info += "\tmodel saved"
                e_count=max_e
            self.logger.info(info)
            print(info)

            if self.scheduler is not None:
                self.scheduler.step()
            
            e_count-=1
            if e_count<1:
                self.logger.info("early stoped")
                break
    
    




