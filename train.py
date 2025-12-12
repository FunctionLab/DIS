## . /mnt/home/yhao1/miniforge3/envs/dis/bin/activate python
## Created by Yun Hao @FunctionLab 2025
## This script trains the DISCO model, a transfer learning model for predicting DIS+ (disease-specific variant impact scores). The script covers both pre-training (ancestry group membership prediction) and fine-tuning step (disease-specific variant impact prediction). For details, see README


## Module
import sys
import argparse
import torch
import numpy as np
import pandas as pd
sys.path.insert(0, 'src/function/')
import data
import train
torch.set_num_threads(20)


## 0. Inputs for training NAME framework  
# Model input 
parser = argparse.ArgumentParser(description = 'Process some integers.')
parser.add_argument('--mode', action = 'store', dest = 'mode', type = str)
parser.add_argument('--device', action = 'store', dest = 'device', type = str, default = 'cpu')
parser.add_argument('--batch_size', action = 'store', dest = 'batch_size', type = int, default = 128)
parser.add_argument('--n_worker', action = 'store', dest = 'n_worker', type = int, default = 4)
parser.add_argument('--out_name', action = 'store', dest = 'out_name', type = str)
# Input specifically for pre-training  
parser.add_argument('--pt_train_info_file', action = 'store', dest = 'pt_train_info_file', type = str)
parser.add_argument('--pt_valid_info_file', action = 'store', dest = 'pt_valid_info_file', type = str)
parser.add_argument('--pt_train_exclude_file', action = 'store', dest = 'pt_train_exclude_file', type = str)
parser.add_argument('--pt_valid_exclude_file', action = 'store', dest = 'pt_valid_exclude_file', type = str)
parser.add_argument('--pt_train_subset_file', action = 'store', dest = 'pt_train_subset_file', type = str, default = 'NA')
parser.add_argument('--pt_valid_subset_file', action = 'store', dest = 'pt_valid_subset_file', type = str, default = 'NA')
parser.add_argument('--pt_n_hidden', action = 'store', dest = 'pt_n_hidden', type = str)
parser.add_argument('--pt_dr', action = 'store', dest = 'pt_dr', type = float)
parser.add_argument('--pt_lr', action = 'store', dest = 'pt_lr', type = float)
parser.add_argument('--pt_l2', action = 'store', dest = 'pt_l2', type = float)
parser.add_argument('--pt_max_epoch', action = 'store', dest = 'pt_max_epoch', type = int, default = 100)
parser.add_argument('--pt_patience', action = 'store', dest = 'pt_patience', type = int, default = 10)
# Input specifically for fine-tuning
parser.add_argument('--ft_train_pos_vep_file', action = 'store', dest = 'ft_train_pos_vep_file', type = str)
parser.add_argument('--ft_train_pos_label_file', action = 'store', dest = 'ft_train_pos_label_file', type = str)
parser.add_argument('--ft_train_neg_vep_file', action = 'store', dest = 'ft_train_neg_vep_file', type = str)
parser.add_argument('--ft_train_neg_label_file', action = 'store', dest = 'ft_train_neg_label_file', type = str)
parser.add_argument('--ft_valid_pos_vep_file', action = 'store', dest = 'ft_valid_pos_vep_file', type = str)
parser.add_argument('--ft_valid_pos_label_file', action = 'store', dest = 'ft_valid_pos_label_file', type = str)
parser.add_argument('--ft_valid_neg_vep_file', action = 'store', dest = 'ft_valid_neg_vep_file', type = str)
parser.add_argument('--ft_valid_neg_label_file', action = 'store', dest = 'ft_valid_neg_label_file', type = str)
parser.add_argument('--ft_relation_file', action = 'store', dest = 'ft_relation_file', type = str)
parser.add_argument('--ft_layer_file', action = 'store', dest = 'ft_layer_file', type = str)
parser.add_argument('--ft_weight_file', action = 'store', dest = 'ft_weight_file', type = str)
parser.add_argument('--ft_weight_pwr', action = 'store', dest = 'ft_weight_pwr', type = float)
parser.add_argument('--ft_ag_info_file', action = 'store', dest = 'ft_ag_info_file', type = str)
parser.add_argument('--ft_pretrain', action = 'store', dest = 'ft_pretrain', type = bool, default = False)
parser.add_argument('--ft_min_module_size', action = 'store', dest = 'ft_min_module_size', type = int)
parser.add_argument('--ft_max_module_size', action = 'store', dest = 'ft_max_module_size', type = int)
parser.add_argument('--ft_n_unfreeze', action = 'store', dest = 'ft_n_unfreeze', type = int, default = 0)
parser.add_argument('--ft_lr', action = 'store', dest = 'ft_lr', type = float)
parser.add_argument('--ft_l2', action = 'store', dest = 'ft_l2', type = float)
parser.add_argument('--ft_mrl_margin', action = 'store', dest = 'ft_mrl_margin', type = float)
parser.add_argument('--ft_mrl_coeff', action = 'store', dest = 'ft_mrl_coeff', type = float)
args = parser.parse_args()

## 1. Pre-training model for ancestry group membership prediction
if args.mode == 'pre-train':
    # Load the variant embedding and ancestry group membership data (in batches of specified size) for model training and validation
    np.random.seed(0)
    pt_train_dict, pt_valid_dict, pt_train_data_loader, pt_valid_data_loader, pt_n_input, pt_n_output = data.load_pretrain_data(
            args.pt_train_info_file, args.pt_valid_info_file, args.pt_train_exclude_file, args.pt_valid_exclude_file, args.pt_train_subset_file, args.pt_valid_subset_file, args.batch_size, args.n_worker)
    # Model pre-training
    torch.manual_seed(0)
    pt_n_hidden_nodes = np.array(args.pt_n_hidden.split(','), dtype = int)
    pt_model, pt_train_summary, pt_valid_loss = train.pretrain_ancestry_group_model(pt_train_data_loader, pt_valid_data_loader, 
            n_input_feat = pt_n_input, 
            n_output_feat = pt_n_output, 
            n_hidden_nodes = pt_n_hidden_nodes, 
            dropout_rate = args.pt_dr, 
            learning_rate = args.pt_lr, 
            l2_lambda = args.pt_l2, 
            patience = args.pt_patience,
            max_epoch = args.pt_max_epoch,
            model_name = args.out_name + '_model.pt',
            model_device = torch.device(args.device))
    # Save model training and validation loss summary for further analysis 
    pt_train_summary.to_csv(args.out_name + '_pt_training_loss_summary.tsv', sep = '\t', index = False, float_format = '%.5f')
    np.savetxt(args.out_name + '_pt_validation_loss.txt', [pt_valid_loss], fmt = '%.5f')    
    pt_train_close = data.process_h5_file('close', file_dict = pt_train_dict)
    pt_valid_close = data.process_h5_file('close', file_dict = pt_valid_dict)


## 2. Fine-tuning model of disease-specific variant impact prediction  
if args.mode == 'fine-tune':
    # Load the pre-trained ancestry group membership prediction model  
    ft_ag_model, pt_n_hidden_nodes = train.load_ancestry_group_model(args.ft_ag_info_file, args.ft_pretrain, args.ft_n_unfreeze)
    # Load the disease ontology hierarchy data to define the configuration of the visible neural network in fine-tuned DIS model  
    ft_parent_dict, ft_root_id, ft_input_module_size, ft_output_module_size, ft_module_pos_weight = data.load_hierarchy_data(args.ft_relation_file, args.ft_layer_file, args.ft_weight_file, pt_n_hidden_nodes[-1], args.ft_min_module_size, args.ft_max_module_size, args.ft_weight_pwr)
    # Load the variant embedding and disease annotation data (in batches of specified size) for model training and validation
    ft_train_pos_data_loader, ft_train_neg_data_loader, ft_valid_pos_data_loader, ft_valid_neg_data_loader = data.load_finetune_data(
            args.ft_train_pos_vep_file, args.ft_train_pos_label_file, args.ft_train_neg_vep_file, args.ft_train_neg_label_file, 
            args.ft_valid_pos_vep_file, args.ft_valid_pos_label_file, args.ft_valid_neg_vep_file, args.ft_valid_neg_label_file, 
            batch_size = args.batch_size, 
            n_workers = args.n_worker)
    # Model fine-tuning 
    torch.manual_seed(0)
    ft_model, ft_train_summary, ft_valid_loss = train.finetune_disease_impact_model(ft_train_pos_data_loader, ft_train_neg_data_loader, ft_valid_pos_data_loader, ft_valid_neg_data_loader, 
            ag_model = ft_ag_model, 
            dis_parent_dict = ft_parent_dict, 
            dis_root = ft_root_id, 
            dis_in_size = ft_input_module_size, 
            dis_out_size = ft_output_module_size, 
            dis_pos_weight = ft_module_pos_weight, 
            learning_rate = args.ft_lr, 
            l2_lambda = args.ft_l2, 
            mrl_margin = args.ft_mrl_margin, 
            mrl_coeff = args.ft_mrl_coeff * len(ft_module_pos_weight),
            model_name = args.out_name + '_model.pt', 
            model_device = torch.device(args.device))
    # Save model training and validation loss summary for further analysis 
    ft_train_summary.to_csv(args.out_name + '_ft_training_loss_summary.tsv', sep = '\t', index = False, float_format = '%.5f')
    np.savetxt(args.out_name + '_ft_validation_loss.txt', [ft_valid_loss], fmt = '%.5f')
