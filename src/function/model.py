## . /mnt/home/yhao1/miniforge3/envs/dis/bin/activate python
## created by Yun Hao @FunctionLab 2025
## This script contains functions to define the model architecture of DISCO model 


## Module
import torch
import torch.nn as nn


## This function defines hidden layer transformation of the pre-training multi-layer perceptron (MLP) model for predicting ancestry group membership
class HiddenLayerModule(nn.Module):
    ## 0. Input arguments
        # M_in (int): number of input neurons
        # M_out (int): number of output neurons

    ## 1. Define module parameters
    def __init__(self, M_in, M_out, D_ratio):
        super(HiddenLayerModule, self).__init__()
        self.linear = nn.Linear(M_in, M_out)
        self.activate = nn.ReLU()
        self.regularize = nn.Dropout(D_ratio) 

    ## 2. Define forward function
    def forward(self, x):
        # Linear transformation
        m_linear = self.linear(x)
        # Nonlinear ReLU activation
        m_active = self.activate(m_linear)
        # Regularization
        m_regularize = self.regularize(m_active)
        return m_regularize


## This function defines ancestry group membership prediction layer (output layer) of the pre-training MLP model
class AGPredictModule(nn.Module):
    ## 0. Input arguments
        # M_in (int): number of input neurons
        # M_out (int): number of ancestry groups to be predicted 

    ## 1. Define module parameters
    def __init__(self, M_in, M_out):
        super(AGPredictModule, self).__init__()
        self.linear = nn.Linear(M_in, M_out)
        self.sigmoid = nn.Sigmoid()
        
    ## 2. Define forward function
    def forward(self, x):
        # Linear transformation
        m_linear = self.linear(x)
        # Group membership probability prediction 
        m_pred = self.sigmoid(m_linear)
        
        return m_pred


## This function defines whole architecture of the pre-training MLP model
class AncestryGroupModel(nn.Module):
    ## 0. Input arguments
        # ip_size (int): number of input variant embedding features  
        # hl_size (1D array): array that contains number of neurons in each hidden layer 
        # op_size (int): number of ancestry groups to be predicted  
        # dp_ratio (float): dropout ratio 

    ## 1. Define module parameters
    def __init__(self, ip_size, hl_size, op_size, dp_ratio):
        super(AncestryGroupModel, self).__init__()
        self.hidden = nn.ModuleList()
        ip_layer = HiddenLayerModule(ip_size, hl_size[0], dp_ratio)
        self.hidden.append(ip_layer) 
        if len(hl_size) > 1:
            for lh in range(0, len(hl_size)-1):
                lh_layer = HiddenLayerModule(hl_size[lh], hl_size[lh+1], dp_ratio)
                self.hidden.append(lh_layer)
        #
        self.output = AGPredictModule(hl_size[-1], op_size)

    ## 2. Define forward function
    def forward(self, x):
        for i, shi in enumerate(self.hidden):
            x = shi(x)
        #
        y = self.output(x)
        return y


## This function defines disease term module transformation of the fine-tuning visible neural network (VNN) model for predicting disease-specific variant impact 
class DisLayerModule(nn.Module):
    ## 0. Input arguments
        # M_in (int): number of input neurons
        # M_out (int): number of output neurons

    ## 1. Define module parameters
    def __init__(self, M_in, M_out):
        super(DisLayerModule, self).__init__()
        self.linear = nn.Linear(M_in, M_out)
        self.activate = nn.ReLU()

    ## 2. Define forward function
    def forward(self, x):
        # Linear transformation
        m_linear = self.linear(x)
        # Nonlinear ReLU activation
        m_active = self.activate(m_linear)
        return m_active


## This function defines impact prediction layer (output layer) of disease term module for the fine-tuning VNN model 
class DisPredictModule(nn.Module):
    ## 0. Input arguments
        # M_in (int): number of input neurons

    ## 1. Define module parameters
    def __init__(self, M_in):
        super(DisPredictModule, self).__init__()
        self.linear = nn.Linear(M_in, 1)
        self.sigmoid = nn.Sigmoid()

    ## 2. Define forward function
    def forward(self, x):
        # Linear transformation
        m_linear = self.linear(x)
        # Disease probability prediction 
        m_sigmoid = self.sigmoid(m_linear)
        return m_sigmoid


## This function defines whole architecture of the fine-tuning VNN model 
class DISModel(nn.Module):
    ## 0. Input arguments
        # pt_model (pytorch model): pre-trained ancestry group memebership prediction model 
        # parent_dict (dictionary): dictionary that contains the parent terms of each disease   
        # root (int): index of VNN root term
        # in_size (1D array): array that contains the computed number of input neurons representing each disease module in the fine-tuning VNN model 
        # out_size (1D array): array that contains the computed number of input neurons representing each disease module in the fine-tuning VNN model  

    ## 1. Define model parameters
    def __init__(self, pt_model, parent_dict, root, in_size, out_size):
        super(DISModel, self).__init__() 
        self.parent_dict = parent_dict
        self.root = root
        self.hidden = list(pt_model.hidden)
        self.layer = nn.ModuleList()
        self.pred = nn.ModuleList()
        for idos, dos in enumerate(out_size):
            self.layer.append(DisLayerModule(in_size[idos], out_size[idos]))
            self.pred.append(DisPredictModule(out_size[idos]))

    ## 2. Define forward function
    def forward(self, x):
        # Forward functions follwing the hidden layers of pre-trained MLP model 
        for i, shi in enumerate(self.hidden):
            x = shi(x)
        # Forward functions follwing the layers of fine-tuning VNN model 
        layer_result = [0] * len(self.layer)
        pred_result = [0] * len(self.layer)
        for lsdl in range(0, len(self.layer)):
            if lsdl in self.root:
                # root disease term takes in the output of final hidden layer of pre-trained MLP model 
                layer_result[lsdl] = self.layer[lsdl](x)
                pred_result[lsdl] = self.pred[lsdl](layer_result[lsdl])
            else:
                # other disease term takes in the concatenated output of its parent terms 
                lsdl_parent = self.parent_dict[lsdl]
                lsdl_parent_output_list = [layer_result[lp] for lp in lsdl_parent]
                lsdl_parent_output = torch.cat(lsdl_parent_output_list, 1)
                # Disease module forward function and impact prediction   
                layer_result[lsdl] = self.layer[lsdl](lsdl_parent_output)
                pred_result[lsdl] = self.pred[lsdl](layer_result[lsdl])
        #
        pred_result1 = torch.cat(pred_result, 1)

        return pred_result1
