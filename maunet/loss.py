import torch
import torch.nn.functional as F

def BCE_Loss(pred,label):
    pred = pred.view(-1)
    label = label.view(-1)
    pred = pred.float()
    label = label.float()
    return F.binary_cross_entropy(pred,label)

def DiceBCE_Loss(inputs, targets, smooth=1):
    
    inputs = inputs.view(-1).float()
    targets = targets.view(-1).float()
    
    intersection = (inputs * targets).sum()                            
    dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)   # 注意这里已经使用1-dice 
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE