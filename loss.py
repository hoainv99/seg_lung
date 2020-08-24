import torch
import torch.nn as nn
import torch.nn.functional as F
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum()
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)))
    
    return loss.mean()
def calc_loss(pred, target, metrics, bce_weight=0.5):
    #     if pred.shape !=(batch_size,224,224):
    #         pred = pred.expand(target.shape[0],224,224)
    bce = F.binary_cross_entropy( pred.float(), target)

    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss