import os
#import matplotlib.pyplot as plt
#import kornia
import torch
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

''' ==================================== Loss ==================================== '''
def L1_loss(score, target):
    loss = torch.mean(torch.abs(score - target))
    return loss

def L2_loss(score, target=1):
    loss = torch.sum((score - target) ** 2, dim=1)
    return loss.mean()

def hinge_loss(score, type='D_real'):
    if type == 'D_real':
        loss = F.relu(1 - score).mean()
    elif type == 'D_fake':
        loss = F.relu(1 + score).mean()
    elif type == 'G':
        loss = -score.mean()
    return loss

# def regression_loss(x, y):
#     x = F.normalize(x, dim=1)
#     y = F.normalize(y, dim=1)
#     return -2 * (x * y).sum(dim=-1)

def byol_loss(pred_1, target_1, pred_2, target_2):
    pred_1 = F.normalize(pred_1, dim=1)
    target_1 = F.normalize(target_1, dim=1)
    pred_2 = F.normalize(pred_2, dim=1)
    target_2 = F.normalize(target_2, dim=1)
    
    loss = -2 * ((pred_1 * target_1).sum(dim=-1) + (pred_2 * target_2).sum(dim=-1))
    return loss.mean()
    

''' ==================================== Function ==================================== '''
def renormalize(x):
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
    x = x / 2 + 0.5
    return (x - mean) / std

