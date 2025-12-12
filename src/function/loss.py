## . /mnt/home/yhao1/miniforge3/envs/dis/bin/activate python
## created by Yun Hao @FunctionLab 2025
## This script contains loss functions used for model training for DISCO model


## Module
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


## This function computes the binary cross entropy (BCE) loss used in pre-training for ancestry group membership prediction   
def ag_bce_loss():
    return nn.BCELoss()


## This function computes the BCE loss, which is part of the loss function used in fine-tuning for disease-specific variant impact prediction  
class DisBCELoss(nn.Module):
    ## 0. Input arguments
        # pos_weight (1D array): array that contains the weight for positive variants in each disease module 

    ## 1. Define parameters
    def __init__(self, pos_weight):
        super(DisBCELoss, self).__init__()
        self.pos_weight = torch.tensor(pos_weight, dtype = torch.float)
        self.neg_weight = 1 - self.pos_weight

    ## 2. Define forward function
    def forward(self, pred_y, y):
        # Compute the normalized weight for each variant  
        pos_weight_full = torch.broadcast_to(self.pos_weight, pred_y.shape)    
        neg_weight_full = torch.broadcast_to(self.neg_weight, pred_y.shape) 
        weight_full = y * pos_weight_full + (1 - y) * neg_weight_full  
        norm_weight = weight_full/weight_full.sum(dim = 0, keepdim = True)
        # Compute weighted loss BCE loss 
        loss_func = nn.BCELoss(weight = norm_weight, reduction = 'sum') 
        m_loss = loss_func(pred_y, y)

        return m_loss


## This function computes the margin rank loss, which is part of the loss function used in fine-tuning for disease-specific variant impact prediction  
class DisMRLoss(nn.Module):
    ## 0. Input arguments
        # margin (float): margin factor 

    ## 1. Define parameters 
    def __init__(self, margin):
        super(DisMRLoss, self).__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin = self.margin, reduction = 'mean')

    ## 2. Define forward function
    def forward(self, pred_y, y): 
        y_sum = torch.sum(y, axis = 1)
        y_sum_full = torch.broadcast_to(y_sum, y.T.shape).T
        y_sum1 = torch.sum(1-y, axis = 1)
        y_sum1_full = torch.broadcast_to(y_sum1, y.T.shape).T
        ys_id = y_sum > 0
        x1_mat = y[ys_id, :] * pred_y[ys_id, :]/y_sum_full[ys_id, :]
        x1_sum = torch.sum(x1_mat, axis = 1)
        x2_mat = (1 - y[ys_id, :]) * pred_y[ys_id, :]/y_sum1_full[ys_id, :]
        x2_sum = torch.sum(x2_mat, axis = 1)
        mr_loss = self.loss(x1_sum, x2_sum, torch.ones(len(x1_sum)))
        
        return mr_loss


## This function computes the overall loss used in fine-tuning for disease-specific variant impact prediction  
class DisLoss(nn.Module):
    ## 0. Input arguments
        # pos_weight (1D array): array that contains the weight for positive variants in each disease module 
        # margin (float): margin factor 
        # coeff (float): coefficient of margin rank loss function 

    ## 1. Define parameters 
    def __init__(self, pos_weight, margin, coeff):
        super(DisLoss, self).__init__()
        self.bceloss = DisBCELoss(pos_weight)
        self.mrloss = DisMRLoss(margin)
        self.coeff = coeff

    ## 2. Define forward function
    def forward(self, pred_y, y):
        bce_loss = self.bceloss(pred_y, y)
        mr_loss = self.mrloss(pred_y, y)  
        total_loss = bce_loss + self.coeff * mr_loss

        return bce_loss, mr_loss, total_loss

