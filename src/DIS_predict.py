## Created by Yun Hao @FunctionLab 2025
## This script runs the fine-tuned DISCO model to predict DIS+ (disease-specific variant impact scores) on levels of both transcriptional regulation (TR) and post-transcriptional regulation (PTR). For details, see README


## Module
import sys
import argparse
import torch
import numpy as np
import pandas as pd
sys.path.insert(0, 'src/function/')
import embedding
import predict
torch.set_num_threads(20)


## 0. Inputs for predicting disease-specific variant impact scores 
parser = argparse.ArgumentParser(description = 'Input for predicting DIS+')
# if input is variant vcf file 
parser.add_argument('--vcf_file', action = 'store', dest = 'vcf_file', type = str, default = 'NA')
parser.add_argument('--hg_version', action = 'store', dest = 'hg_version', type = str, default = 'NA')
parser.add_argument('--method', action = 'store', dest = 'method', type = str)  # 'Sei' for TR model, 'Seqweaver' for PTR model
parser.add_argument('--out_name', action = 'store', dest = 'out_name', type = str)
# if input is variant effect prediction (embedding) h5 file 
parser.add_argument('--vep_file', action = 'store', dest = 'vep_file', type = str)
# running configurations
parser.add_argument('--device', action = 'store', dest = 'device', type = str, default = 'cpu')
parser.add_argument('--n_worker', action = 'store', dest = 'n_worker', type = int, default = 4)
parser.add_argument('--batch_size', action = 'store', dest = 'batch_size', type = int, default = 128)
parser.add_argument('--n_repeat', action = 'store', dest = 'n_repeat', type = int, default = 20)
parser.add_argument('--start_id', action = 'store', dest = 'start_id', type = int, default = 0)
parser.add_argument('--end_id', action = 'store', dest = 'end_id', type = int, default = -1)
# model hyperparamteres
parser.add_argument('--setting', action = 'store', dest = 'setting', type = int, default = 1)   # 0 for user-specified, 1 for default setting 
parser.add_argument('--ft_relation_file', action = 'store', dest = 'ft_relation_file', type = str)
parser.add_argument('--ft_layer_file', action = 'store', dest = 'ft_layer_file', type = str)
parser.add_argument('--ft_ag_info_file', action = 'store', dest = 'ft_ag_info_file', type = str)
parser.add_argument('--ft_pretrain', action = 'store', dest = 'ft_pretrain', type = bool, default = True)
parser.add_argument('--ft_min_module_size', action = 'store', dest = 'ft_min_module_size', type = int)
parser.add_argument('--ft_max_module_size', action = 'store', dest = 'ft_max_module_size', type = int)
parser.add_argument('--ft_n_unfreeze', action = 'store', dest = 'ft_n_unfreeze', type = int)
parser.add_argument('--ft_model', action = 'store', dest = 'ft_model', type = str)
args = parser.parse_args()
   
## 1. Set the default model hyperparamteres
if args.setting == 1:
    if args.method == 'Sei':
        args.ft_relation_file = 'model/tr_dis/DO_tr_dis_node_parent.tsv' 
        args.ft_layer_file = 'model/tr_dis/DO_tr_dis_node_layer.tsv'
        args.ft_ag_info_file = 'model/tr_dis/tr_dis_pretrain_info.txt'
        args.ft_min_module_size = 10
        args.ft_max_module_size = 100
        args.ft_n_unfreeze = 0
        args.ft_model = 'model/tr_dis/tr_dis_finetune.pt'
    elif args.method == 'Seqweaver':
        args.ft_relation_file = 'model/ptr_dis/DO_ptr_dis_node_parent.tsv'
        args.ft_layer_file = 'model/ptr_dis/DO_ptr_dis_node_layer.tsv'
        args.ft_ag_info_file = 'model/ptr_dis/ptr_dis_pretrain_info.txt'
        args.ft_min_module_size = 10
        args.ft_max_module_size = 100
        args.ft_n_unfreeze = 0
        args.ft_model = 'model/ptr_dis/ptr_dis_finetune.pt'
    else:
        raise ValueError("Unknown --method. Use 'Sei' or 'Seqweaver'.")

## 2. Generate variant embedding h5 file if input is variant vcf file
if args.vcf_file != 'NA' and args.hg_version != 'NA':
    args.vep_file = embedding.compute_variant_embedding(args.vcf_file, args.out_name, args.hg_version, args.method, args.device)
if not args.vep_file:
    raise ValueError("Provide --vep_file or (vcf_file + hg_version).")

## 3. Prepare for model loading, read in the model configurations including the size of pre-trained layers and fine-tuned layers 
ft_ag_model, pt_n_hidden_nodes = predict.load_ancestry_group_model(args.ft_ag_info_file, args.ft_pretrain, args.ft_n_unfreeze)
ft_parent_dict, ft_root_id, ft_input_module_size, ft_output_module_size = predict.load_hierarchy_data(args.ft_relation_file, args.ft_layer_file, pt_n_hidden_nodes[-1], args.ft_min_module_size, args.ft_max_module_size)

## 4. Load the generated variant embedding and implement model to predict disease-specific variant impact scores 
pred_data_loader = predict.load_predict_data(args.vep_file, start_id = args.start_id, end_id = args.end_id, b_size = args.batch_size, n_workers = args.n_worker)
out_pred_name = args.out_name + '_' + args.method + '_dis_pred_' + str(pred_data_loader.dataset.start_index) + '_' + str(pred_data_loader.dataset.end_index) + '.h5'
torch.manual_seed(0)
model_pred = predict.predict_disease_impact(pred_data_loader, 
        ag_model = ft_ag_model, 
        dis_parent_dict = ft_parent_dict, 
        dis_root = ft_root_id, 
        dis_in_size = ft_input_module_size, 
        dis_out_size = ft_output_module_size, 
        model_name = args.ft_model,
        out_name = out_pred_name,
        num_repeat = args.n_repeat,
        model_device = torch.device(args.device))
