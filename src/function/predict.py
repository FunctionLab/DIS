## . /mnt/home/yhao1/miniforge3/envs/dis/bin/activate python
## created by Yun Hao @FunctionLab 2025
## This script contains functions used for running fine-tuned DISCO model to predict DIS+ (disease-specific variant impact scores) 


## Module
import sys
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
sys.path.insert(0, 'src/function/')
import model


## This function defines the variant embedding data object 
class DISPredictSet(Dataset):
    ## 0. Input arguments
        # feat_file (str): path of feature embedding file ('_diff.h5' file from running Seqweaver or Sei)  
        # start_index (int): lower bound index (based on the data object from h5 file) for variants to be predicted
        # end_index (int): upper bound index (based on the data object from h5 file) for variants to be predicted

    def __init__(self, feat_file, start_index, end_index):
        super(DISPredictSet, self).__init__()
        self.feat_h5 = h5py.File(feat_file, 'r')
        self.start_index = start_index
        if end_index >= 0:
            self.end_index = end_index
        else:
            self.end_index = len(self.feat_h5['data']) + end_index + 1
    
    def __getitem__(self, index):
        feature = torch.tensor(np.abs(self.feat_h5['data'][index+self.start_index]), dtype = torch.float)
        return feature
    
    def __len__(self):
        return(self.end_index - self.start_index)


## This function loads the variant embedding data used for disease-specific variant impact prediction 
def load_predict_data(vep_file, start_id = 0, end_id = -1, b_size = 128, n_workers = 4):
    ## 0. Input arguments
        # vep_file (str): path of feature embedding file ('_diff.h5' file from running Seqweaver or Sei)  
        # start_id (int): lower bound index (based on vep h5 file) for variants to be predicted (Default: 0) 
        # end_id (int): upper bound index (based on vep h5 file) for variants to be predicted (Default: -1) 
        # b_size (int): batch size (Default: 128)
        # n_workers (int): number of workers (Default: 4)

    predict_data = DISPredictSet(vep_file, start_id, end_id)
    N_predict = predict_data.__len__()
    predict_data_loader = DataLoader(predict_data, batch_size = b_size, num_workers = n_workers)
    
    return predict_data_loader


## This function computes the number of input/output neurons that represent each disease module in the visible neural network (VNN) of fine-tuned DIS model   
def compute_disease_module_stat(layer_df, parent_dict, root_id, root_input_size, min_m_size, max_m_size):
    ## 0. Input arguments
        # layer_df (pandas DataFrame): data frame that contains the layer number of disease terms  
        # parent_dict (dictionary): data frame that contains the parent terms of each disease term 
        # root_id (int): index of Disease Ontology root term in layer_df
        # root_input_size (int): number of neurons in the last hidden layer of pre-trained model (i.e. input size for the root Diseae module/first layer of VNN model) 
        # min_m_size (int): mininum disease module size
        # max_m_size (int): maximum disease module size

    ## 1. Compute the number of output neurons based on the layer number of each disease module 
    n_max_layer = layer_df.layer_number.max()
    layer_n = layer_df.layer_number.values
    output_size = max_m_size - layer_n/n_max_layer * (max_m_size - min_m_size)
    output_size = np.ceil(output_size).astype(int)

    ## 2. Compute the number of input neurons based on the parent/child relationship between disease terms (input size = sum of output size of parent terms)
    input_size = np.zeros(layer_df.shape[0], dtype = int)
    for lnv in range(layer_df.shape[0]):
        if lnv in root_id:
            input_size[lnv] = root_input_size
        else:
            input_size[lnv] = np.sum(output_size[parent_dict[lnv]])

    return input_size, output_size


## The function loads the disease ontology hierarchy data to define the configuration of the VNN in fine-tuned DIS model  
def load_hierarchy_data(relation_file, layer_file, root_input_size, min_m_size, max_m_size):
    ## 0. Input arguments
        # relation_file (str): path of file that contains parent/children disease relationships 
        # layer_file (str): path of file that contains layer number of nodes
        # root_input_size (int): number of neurons in the last hidden layer of pre-trained model (i.e. input size for the root Diseae module/first layer of VNN model) 
        # min_m_size (int): mininum disease module size
        # max_m_size (int): maximum disease module size

    ## 1. Obtain parent terms of each disease term, store information in a dictionary
    relation_df = pd.read_csv(relation_file, sep = '\t')
    parent_dict = {}
    for i in range(0, relation_df.shape[0]):
        i_child = relation_df.child_node_id.values[i]
        i_parent = relation_df.parent_node_id.values[i].split(',')
        parent_dict[i_child] = np.array(i_parent).astype(int)

    ## 2. Compute the number of input/output neurons that represent each disease module in the VNN of fine-tuned DIS model 
    layer_df = pd.read_csv(layer_file, sep = '\t')
    root_id = layer_df[layer_df.layer_number == 0].node_id.values
    input_module_size, output_module_size = compute_disease_module_stat(layer_df, parent_dict, root_id, root_input_size, min_m_size, max_m_size)

    return parent_dict, root_id, input_module_size, output_module_size


## This function loads the pre-trained ancestry group membership prediction model 
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


## This function implement fine-tuned disease impact prediction model to predict diseae-specific variant impact scores 
def predict_disease_impact(pred_data_loader, ag_model, dis_parent_dict, dis_root, dis_in_size, dis_out_size, model_name, out_name, num_repeat, model_device):
    ## 0. Input argument
        # pred_data_loader (torch DataLoader): loader that contains the variant embedding data used for disease-specific variant impact prediction  
        # ag_model (torch model): loaded pre-trained ancestry group memebership prediction model 
        # dis_parent_dict (dictionary): dictionary that contains the parent terms of each disease  
        # dis_root (int): index of VNN root term
        # dis_in_size (1D array): array that contains the computed number of input neurons representing each disease module in the VNN of fine-tuned DIS model
        # dis_out_size (1D array): array that contains the computed number of output neurons representing each disease module in the VNN of fine-tuned DIS model
        # model_name (str): path of fine-tuned DIS model '.pt' file  
        # out_name (str): path of prediction '.h5' file storing the prediction results 
        # num_repeat (int): number of times to implement the fine-tuned for prediction, the eventual output result for each variant is the average across all implementations (result may vary slightly among implementations due to the limitation of float32/float64)
        # model_device (torch device): torch object for computing device
        
    ## 1. Load the fine-tuned disease impact prediction model 
    dis_model = model.DISModel(ag_model, dis_parent_dict, dis_root, dis_in_size, dis_out_size)
    model_state = torch.load(model_name)
    dis_model.load_state_dict(model_state['model_state_dict'])
    dis_model.to(model_device)
    dis_model.eval()

    ## 2. Implement loaded fine-tuned disease impact prediction model to predict variant impact, write results output file 
    start_index = 0
    with h5py.File(out_name, 'a') as out_h5:
        out_h5.create_dataset("data", (len(pred_data_loader.dataset), len(dis_in_size)), dtype = 'float32')
        out_data = out_h5["data"]
        for j, batch_data in enumerate(pred_data_loader, 0):
            batch_feature = batch_data
            batch_pred_rep = torch.zeros(num_repeat, batch_feature.shape[0], len(dis_in_size))
            for i in range(0, num_repeat):
                batch_pred_rep[i, :, :] = dis_model(batch_feature)
            batch_pred = torch.mean(batch_pred_rep, dim = 0)
            out_data[start_index :(start_index + batch_pred.shape[0])] = batch_pred.cpu().detach().numpy().copy()
            start_index = start_index + batch_pred.shape[0]

    return 1 
