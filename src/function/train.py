## . /mnt/home/yhao1/miniforge3/envs/dis/bin/activate python
## created by Yun Hao @FunctionLab 2025
## This script contains functions used for model training for DISCO model


## Module
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, 'src/function/')
import early_stop
import loss
import model


## This function pre-trains a multi-layer perceptron (MLP) model for predicting variant ancestry group membership  
def pretrain_ancestry_group_model(train_data_loader, valid_data_loader, n_input_feat, n_output_feat, n_hidden_nodes, dropout_rate, learning_rate, l2_lambda, patience = 10, max_epoch = 100, model_name = 'checkpoint.pt', model_device = 'cpu'):
    ## 0. Input argument
        # train_data_loader (torch DataLoader): data loader that contains the formatted embedding and ancestry group annotation for training variants 
        # valid_data_loader (torch DataLoader): data loader that contains the formatted embedding and ancestry group annotation for validation variants 
        # n_input_feat (int): number of input features (variant embeddings)
        # n_output_feat (int): number of ancestry groups to be predicted 
        # n_hidden_nodes (1D array): number of neurons in each hidden layer 
        # dropout_rate (float): dropout rate
        # learning_rate (float): learning rate
        # l2_lambda (float): L2 regularization factor of loss function
        # patience (int): maximal number of epochs to run after testing loss stop decreasing, i.e. optimal model is reached (Default: 10) 
        # max_epoch (int): maximal number of epochs to run before stopping if an optimal testing loss cannot be reached (Default: 100)
        # model_name (str): output file storing trained model (Default: 'checkpoint.pt')
        # model_device (torch device): device to perform training (Default: 'cpu')

    ## 1. Define MLP model
    # MLP model architecture 
    ag_model = model.AncestryGroupModel(n_input_feat, n_hidden_nodes, n_output_feat, dropout_rate)
    ag_model.to(model_device)
    # Adam optimizer
    ag_optim = torch.optim.Adam(ag_model.parameters(), lr = learning_rate, weight_decay = l2_lambda)
    lambda1 = lambda ep: (10 ** (-3/(max_epoch-1))) ** ep
    ag_sched = torch.optim.lr_scheduler.LambdaLR(ag_optim, lr_lambda = lambda1)
    # Define early stop function
    ag_stop = early_stop.stop(patience = patience, model_name = model_name)
    ag_loss = loss.ag_bce_loss()

    ## 2. Perform model training 
    epoch = 0
    train_loss = []
    valid_loss = []
    while ag_stop.early_stop == False:
        epoch += 1
        # Set model to training mode
        ag_model.train()
        # Iterate by mini-batch, perform forward and backward propogation, keep track of training/validation loss
        ep_train_loss = np.zeros(len(train_data_loader))
        for i, batch_data in enumerate(train_data_loader, 0):
            # Get variant embedding and ancestry group annotation data of current batch
            batch_feature, batch_label = batch_data[0].to(model_device), batch_data[1].to(model_device)
            # Set the gradients to 0
            ag_optim.zero_grad()
            # Perform forward propogation to compute predicted output
            batch_pred = ag_model(batch_feature)
            # Compute loss of current batch
            batch_loss = ag_loss(batch_pred, batch_label)
            ep_train_loss[i] = float(batch_loss.cpu().detach().data)
            # Perform backward propogation
            batch_loss.backward()
            # Perform optimization
            ag_optim.step()
        # Average computed training loss over all mini-batches, store the average
        train_loss.append(np.mean(ep_train_loss))
        np.savetxt(model_name + '_training_loss_epoch_' + str(epoch) + '.txt', ep_train_loss, fmt = '%.5f')
        # Implement current model to validation data, perform forward propogation to compute predicted output
        ag_model.eval()
        ep_valid_loss = np.zeros(len(valid_data_loader))
        for j, batch_data in enumerate(valid_data_loader, 0):
            # Get variant embedding and ancestry group annotation data of current batch
            batch_feature, batch_label = batch_data[0].to(model_device), batch_data[1].to(model_device)
            # Perform forward propogation to compute predicted output
            batch_pred = ag_model(batch_feature)
            # Compute loss of current batch
            batch_loss = ag_loss(batch_pred, batch_label)
            ep_valid_loss[j] = float(batch_loss.cpu().detach().data)
        # Average computed validation loss over all mini-batches, store the average
        valid_loss.append(np.mean(ep_valid_loss))
        np.savetxt(model_name + '_validation_loss_epoch_' + str(epoch) + '.txt', ep_valid_loss, fmt = '%.5f')
        # Check if early stop criterion has been met
        ag_stop(valid_loss[-1], ag_model, ag_optim)
        # If so, load the last checkpoint with the best model
        if ag_stop.early_stop:
            stop_point_state = torch.load(model_name)
            ag_model.load_state_dict(stop_point_state['model_state_dict'])
            ag_optim.load_state_dict(stop_point_state['optimizer_state_dict'])
            # Store training and validation loss of every epoch in data frame format
            train_epoch = np.arange(1, epoch+1)
            train_summary = pd.DataFrame({'epoch': train_epoch, 'training_loss': train_loss,  'validation_loss': valid_loss})
            final_valid_loss = train_summary.validation_loss.values[-patience-1]
            break
        # Stop training if the maximum epoch is reached
        if epoch == max_epoch:
            train_epoch = np.arange(1, epoch+1)
            train_summary = pd.DataFrame({'epoch': train_epoch, 'training_loss': train_loss,  'validation_loss': valid_loss})
            final_valid_loss = train_summary.validation_loss.values[-1]
            break
        #
        ag_sched.step()

    return ag_model, train_summary, final_valid_loss


##  This function loads the pre-trained ancestry group membership prediction model 
def load_ancestry_group_model(ag_info_file, pre_train = False, n_unfreeze = 0):
    ## 0. Input argument
        # ag_info_file (str): path of file that contains the configuration info of ancestry group membership prediction model 
        # pre_train (bool): whether to use the model weight parameters learned from pre-training (Default: False)
        # n_unfreeze (int): number of hidden layers in the pre-trained model to unfreeze for fine-tuning (Default: 0)

    ## 1. Read in lines of the configuration info file, store info in dictionary 
    info_f = open(ag_info_file, 'r')
    info_lines = info_f.readlines()
    ag_info_dict = {}
    for il in info_lines:
        il_s1 = il.strip()
        il_s2 = il_s1.split(': ')
        ag_info_dict[il_s2[0]] = il_s2[1]

    ## 2. Parse the configuration info
    n_input_feat = int(ag_info_dict['Number of input features'])
    n_output_feat = int(ag_info_dict['Number of output features'])
    n_hidden_nodes = np.array(ag_info_dict['Number of hidden neurons'].split(','), dtype = int)
    dr = float(ag_info_dict['Dropout rate'])
    ag_model = model.AncestryGroupModel(n_input_feat, n_hidden_nodes, n_output_feat, dr)

    ## 3. Load the learned weight parameters of pre-trained model if specified so 
    if pre_train:
        model_state = torch.load(ag_info_dict['Optimal pretrain model'])
        ag_model.load_state_dict(model_state['model_state_dict'])
        # Freeze the layers based on the 'n_unfreeze'
        n_hidden_layer = len(n_hidden_nodes)
        n_freeze = n_hidden_layer - n_unfreeze
        if n_freeze > 0:
            for nf in range(0, n_freeze):
                for param in ag_model.hidden[nf].parameters():
                    param.requires_grad = False

    return ag_model, n_hidden_nodes


## This function fine-tunes the pre-trained model by incorporating a visible neural network (VNN) model for predicting disease-specific variant impact
def finetune_disease_impact_model(train_pos_data_loader, train_neg_data_loader, valid_pos_data_loader, valid_neg_data_loader, ag_model, dis_parent_dict, dis_root, dis_in_size, dis_out_size, dis_pos_weight, learning_rate, l2_lambda, mrl_margin, mrl_coeff, patience = 10, max_epoch = 100, model_name = 'checkpoint.pt', model_device = 'cpu'):
    ## 0. Input argument
        # train_pos_data_loader (torch DataLoader): data loader that contains the formatted embedding and disease annotation for positive (pathogenic) variants in training set
        # train_neg_data_loader (torch DataLoader): data loader that contains the formatted embedding and disease annotation for negative (neutral) variants in training set
        # valid_pos_data_loader (torch DataLoader): data loader that contains the formatted embedding and disease annotation for positive (pathogenic) variants in validation set
        # valid_neg_data_loader (torch DataLoader): data loader that contains the formatted embedding and disease annotation for negative (neutral) variants in validation set
        # ag_model (torch model): loaded pre-trained ancestry group memebership prediction model 
        # dis_parent_dict (dictionary): dictionary that contains the parent terms of each disease  
        # dis_root (int): index of VNN root term 
        # dis_in_size (1D array): array that contains the computed number of input neurons representing each disease module in the VNN of fine-tuned DIS model
        # dis_out_size (1D array): array that contains the computed number of output neurons representing each disease module in the VNN of fine-tuned DIS model
        # dis_pos_weight (1D array): array that contains the weight for positive variants in each disease module to be used for computing loss function 
        # learning_rate (float): learning rate 
        # l2_lambda (float): L2 regularization factor of loss function 
        # mrl_margin (float): margin factor to be used for computing margin rank loss function  
        # mrl_coeff (float): coefficient of margin rank loss function  
        # patience (int): maximal number of epochs to run after testing loss stop decreasing, i.e. optimal model is reached (Default: 10) 
        # max_epoch (int): maximal number of epochs to run before stopping if an optimal testing loss cannot be reached (Default: 100)
        # model_name (str): output file storing trained model (Default: 'checkpoint.pt')
        # model_device (torch device): device to perform training (Default: 'cpu')

    ## 1. Define DIS model
    # DIS model architecture
    dis_model = model.DISModel(ag_model, dis_parent_dict, dis_root, dis_in_size, dis_out_size)
    dis_model.to(model_device)
    # Adam optimizer
    dis_optim = torch.optim.Adam(dis_model.parameters(), lr = learning_rate, weight_decay = l2_lambda)
    lambda1 = lambda ep: (10 ** (-3/(max_epoch-1))) ** ep
    dis_sched = torch.optim.lr_scheduler.LambdaLR(dis_optim, lr_lambda = lambda1)
    # Define early stop function
    dis_stop = early_stop.stop(patience = patience, model_name = model_name)
    dis_loss = loss.DisLoss(dis_pos_weight, mrl_margin, mrl_coeff)

    ## 2. Perform model training 
    epoch = 0
    train_bce_loss = []
    train_mr_loss = []
    train_total_loss = []
    valid_bce_loss = []
    valid_mr_loss = []
    valid_total_loss = []
    while dis_stop.early_stop == False:
        epoch += 1
        # Set model to training mode
        dis_model.train()
        # Iterate by mini-batch, perform forward and backward propogation, keep track of training/validation loss
        ep_train_bce_loss = np.zeros(len(train_pos_data_loader))
        ep_train_mr_loss = np.zeros(len(train_pos_data_loader))
        ep_train_total_loss = np.zeros(len(train_pos_data_loader))
        for i, batch_data in enumerate(zip(train_pos_data_loader, train_neg_data_loader), 0):
            # Get embedding and disease annotation data of current batch
            batch_feature = torch.cat((batch_data[0][0], batch_data[1][0]), 0).to(model_device)
            batch_label = torch.cat((batch_data[0][1], batch_data[1][1]), 0).to(model_device)
            # Set the gradients to 0
            dis_optim.zero_grad()
            # Perform forward propogation to compute predicted output
            batch_pred = dis_model(batch_feature)
            # Compute bce loss, margin rank Loss, and overall loss of current batch
            batch_bce_loss, batch_mr_loss, batch_total_loss = dis_loss(batch_pred, batch_label)
            ep_train_bce_loss[i] = float(batch_bce_loss.cpu().detach().data)
            ep_train_mr_loss[i] = float(batch_mr_loss.cpu().detach().data)
            ep_train_total_loss[i] = float(batch_total_loss.cpu().detach().data)
            # Perform backward propogation
            batch_total_loss.backward()
            # Perform optimization
            dis_optim.step()
        # Average computed training loss over all mini-batches, store the average
        train_bce_loss.append(np.mean(ep_train_bce_loss))
        train_mr_loss.append(np.mean(ep_train_mr_loss))
        train_total_loss.append(np.mean(ep_train_total_loss))
        # Implement current model to validation data, perform forward propogation to compute predicted output
        dis_model.eval()
        ep_valid_bce_loss = np.zeros(len(valid_pos_data_loader))
        ep_valid_mr_loss = np.zeros(len(valid_pos_data_loader))
        ep_valid_total_loss = np.zeros(len(valid_pos_data_loader))
        for j, batch_data in enumerate(zip(valid_pos_data_loader, valid_neg_data_loader), 0):
            # Get embedding and disease annotation data of current batch
            batch_feature = torch.cat((batch_data[0][0], batch_data[1][0]), 0).to(model_device)
            batch_label = torch.cat((batch_data[0][1], batch_data[1][1]), 0).to(model_device)
            # Perform forward propogation to compute predicted output
            batch_pred = dis_model(batch_feature)
            # Compute bce loss, margin rank Loss, and overall loss of current batch
            batch_bce_loss, batch_mr_loss, batch_total_loss = dis_loss(batch_pred, batch_label)
            ep_valid_bce_loss[j] = float(batch_bce_loss.cpu().detach().data)
            ep_valid_mr_loss[j] = float(batch_mr_loss.cpu().detach().data)
            ep_valid_total_loss[j] = float(batch_total_loss.cpu().detach().data)
        # Average computed validation loss over all mini-batches, store the average
        valid_bce_loss.append(np.mean(ep_valid_bce_loss))
        valid_mr_loss.append(np.mean(ep_valid_mr_loss))
        valid_total_loss.append(np.mean(ep_valid_total_loss))
        print(epoch)
        # Check if early stop criterion has been met
        dis_stop(valid_total_loss[-1], dis_model, dis_optim)
        # If so, load the last checkpoint with the best model
        if dis_stop.early_stop:
            stop_point_state = torch.load(model_name)
            dis_model.load_state_dict(stop_point_state['model_state_dict'])
            dis_optim.load_state_dict(stop_point_state['optimizer_state_dict'])
            # Store training and validation loss of every epoch in data frame format
            train_epoch = np.arange(1, epoch+1)
            train_summary = pd.DataFrame({'epoch': train_epoch, 'training_bce_loss': train_bce_loss,  'training_mr_loss': train_mr_loss, 'training_total_loss': train_total_loss, 'validation_bce_loss': valid_bce_loss, 'validation_mr_loss': valid_mr_loss, 'validation_total_loss': valid_total_loss})
            final_valid_loss = train_summary.validation_total_loss.values[-patience-1]
            break
        # Stop training if the maximum epoch is reached
        if epoch == max_epoch:
            train_epoch = np.arange(1, epoch+1)
            train_summary = pd.DataFrame({'epoch': train_epoch, 'training_bce_loss': train_bce_loss,  'training_mr_loss': train_mr_loss, 'training_total_loss': train_total_loss, 'validation_bce_loss': valid_bce_loss, 'validation_mr_loss': valid_mr_loss, 'validation_total_loss': valid_total_loss})
            final_valid_loss = train_summary.validation_total_loss.values[-1]
            break
        #
        dis_sched.step()

    return dis_model, train_summary, final_valid_loss

