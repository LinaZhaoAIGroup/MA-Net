import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

__EPS__ = 1e-9

# mIoU
def get_miou(pred,mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    TP = ((pred==1)&(mask==1)).sum()
    FP = ((pred==1)&(mask==0)).sum()
    TN = ((pred==0)&(mask==0)).sum()
    FN = ((pred==0)&(mask==1)).sum()
    miou = 0.5*(TP/(FN+FP+TP+__EPS__) + TN/(FN+FP+TN+__EPS__))
    return miou

# Dice
def get_dice(pred,mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    TP = ((pred==1)&(mask==1)).sum()
    FP = ((pred==1)&(mask==0)).sum()
    FN = ((pred==0)&(mask==1)).sum()
    
    dice = (2*TP)/(2*TP+FP+FN+__EPS__)
    return dice

# Precision
def get_pre(pred,mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    TP = ((pred==1)&(mask==1)).sum()
    FP = ((pred==1)&(mask==0)).sum()
    
    pre = TP/(TP+FP+__EPS__)
    return pre

# Accuracy
def get_acc(pred,mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    TP = ((pred==1)&(mask==1)).sum()
    FP = ((pred==1)&(mask==0)).sum()
    TN = ((pred==0)&(mask==0)).sum()
    FN = ((pred==0)&(mask==1)).sum()
    
    acc = (TP+TN)/(TP+FP+TN+FN+__EPS__)
    return acc



# 计算模型参数量
def model_param_count(model):
    return sum([p.numel() for p in model.parameters()])


# 展示分割结果    img: 原图像(3,512,512); mask: 原mask图像(512,512); pred_mask: 分割结果(512,512) 
def result_show(img,mask,pred_mask):
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.title("img")
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.title("mask")
    plt.imshow(mask,cmap="RdYlBu_r")
    plt.subplot(1,3,3)
    plt.title("pred")
    plt.imshow(pred_mask,cmap="RdYlBu_r")
    plt.show()
    
