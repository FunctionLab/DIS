## . /mnt/home/yhao1/miniforge3/envs/dis/bin/activate python
## created by Yun Hao @FunctionLab 2025
## This script contains functions used for data loading of DISCO model 


## Module
import h5py
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from typing import Iterator, List


## This function opens/closes variant embedding and ancestry group annotation data h5 file  
def process_h5_file(mode, file_map = None, file_dict = {}):
    ## 0. Input arguments
        # mode (str): 'open' or 'close' 
        # file_map (pandas DataFrame): data frame that contains the mapping between variant embedding file and ancestry group annotation file 
        # file_dict (dictionary): dictionary that contains all processed h5 data object    

    ## 1. Open mode
    if mode == 'open':
        for fvv in file_map.vep_file.values:
            file_dict[fvv] = h5py.File(fvv, 'r')
        for fav in file_map.af_file.values:
            file_dict[fav] = h5py.File(fav, 'r')
        return file_dict

    ## 2. Close mode
    if mode == 'close':
        for key, value in file_dict.items():
            file_dict[key].close()
        return 1


## This function converts a continuous allele frequency vector to binary vector representing ancestry group membership 
def encode_ancestry_group(af_vec, ag_max = [41610, 15314, 3472, 5208, 10632, 316, 68058, 4838]):
    ## 0. Input arguments
        # af_vec (1D array): array that contains the variant allele frequency across ancestry groups 
        # ag_max (1D array): array that contains the number of samples in each ancestry 

    af_cut = 1/np.array(ag_max) * 0.1
    ag_ec = np.array(af_vec > af_cut, dtype = float)

    return ag_ec


## This function load the variant embedding and ancestry group membership data of given variant
def _load_vep_ag_data(file_dict, vep_file, af_file, var_id):
    ## 0. Input arguments
        # file_dict (dictionary): dictionary that contains all processed h5 data object
        # vep_file (str): name of the variant embedding file for the given variant
        # af_file (str): name of the ancestry group annotation file for the given variant
        # var_id (int): index of given variant in the variant embedding/ancestry group annotation data object 

    vep_data = torch.tensor(np.abs(file_dict[vep_file]['data'][var_id]), dtype = torch.float)
    ag_data = torch.tensor(encode_ancestry_group(file_dict[af_file]['data'][var_id][1:]), dtype = torch.float)

    return vep_data, ag_data


## This function defines variant sampler used for model pre-training  
class AGBatchSampler(Sampler[List[int]]):
    ## 0. Input arguments 
        # N_batch_size (int): batch size for sampling 
        # valid_id (1D array): array that contains the index of variants to sample from
        # shuffle (bool): bool that indicates whether shuffle the valid_id before sampling  

    def __init__(self, N_batch_size, valid_id, shuffle) -> None:
        self.N_batch_size = N_batch_size
        self.valid_id = valid_id
        self.shuffle = shuffle
        self.N_valid = len(self.valid_id)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            np.random.shuffle(self.valid_id)
        for lower_bound in range(0, self.N_valid, self.N_batch_size):
            if lower_bound + self.N_batch_size < self.N_valid:
                upper_bound = lower_bound + self.N_batch_size
            else:
                upper_bound = self.N_valid
            batch_id = self.valid_id[lower_bound:upper_bound]
            yield batch_id

    def __len__(self) -> int:
        return (self.N_valid + self.N_batch_size - 1)//self.N_batch_size


## This function defines format for variant embedding and ancestry group membership data used for model pre-training 
class AGDataSet(Dataset):
    ## 0. Input arguments
        # file_dict (dictionary): dictionary that contains all processed h5 data object
        # file_map (pandas DataFrame): data frame that contains the mapping between variant embedding file and ancestry group annotation file 
        # id_interval (1D array): array that contains the accumulative number of variants from 1st to last embedding/annotation file
        # N_valid (int): number of variants to sample from  

    def __init__(self, file_dict, file_map, id_interval, N_valid):
        super(AGDataSet, self).__init__()
        self.file_dict = file_dict
        self.file_map = file_map
        self.id_interval = id_interval
        self.N_valid = N_valid  
    
    def __getitem__(self, index):
        index_diff = index - self.id_interval 
        file_id = np.where(index_diff < 0, index_diff, -np.inf).argmax()
        if file_id == 0:
            var_id = index
        else:
            var_id = index - self.id_interval[file_id-1]
        feature, label = _load_vep_ag_data(self.file_dict, self.file_map.vep_file.values[file_id], self.file_map.af_file.values[file_id], var_id)
        return feature, label
    
    def __len__(self):
        return self.N_valid


## This function loads the variant embedding and ancestry group membership data for model pre-training 
def load_pretrain_data(train_info_file, valid_info_file, train_exclude_file = 'NA', valid_exclude_file = 'NA', train_subset_file = 'NA', valid_subset_file = 'NA', batch_size = 128, n_workers = 4):
    ## 0. Input arguments
        # train_info_file (str): path of file that contains embedding-ancestry group membership file map used for training
        # valid_info_file (str): path of file that contains embedding-ancestry group membership file map used for validation 
        # train_exclude_file (str): path of file that contains the index of variants to be excluded from training (Default: 'NA') 
        # valid_exclude_file (str): path of file that contains the index of variants to be excluded from validation (Default: 'NA') 
        # train_subset_file (str): path of file that contains the index of subset variants to be used for training (Default: 'NA') 
        # valid_subset_file (str): path of file that contains the index of subset variants to be used for training (Default: 'NA') 
        # batch_size (int): batch size (Default: 128)
        # n_workers (int): number of workers (Default: 4)
        
    ## 1. Read in the index of variants to be excluded from training and validation, if any 
    train_exclude_ids = []
    if train_exclude_file != 'NA':
        train_exclude_ids = np.loadtxt(train_exclude_file, dtype = int)
    valid_exclude_ids = []
    if valid_exclude_file != 'NA':
        valid_exclude_ids = np.loadtxt(valid_exclude_file, dtype = int)

    ## 2. Load the embedding and ancestry group membership data for variants used for training 
    train_info_df = pd.read_csv(train_info_file, sep = '\t', header = 0)
    train_id_interval = np.cumsum(train_info_df.n_var.values)
    if train_subset_file != 'NA':
        train_subset_ids = np.loadtxt(train_subset_file, dtype = int)
        final_train_ids = np.setdiff1d(train_subset_ids, train_exclude_ids)
    else:
        final_train_ids = np.setdiff1d(np.arange(train_id_interval[-1]), train_exclude_ids)
    #
    train_file_dict = process_h5_file('open', file_map = train_info_df)
    train_data = AGDataSet(train_file_dict, train_info_df, train_id_interval, len(final_train_ids))
    example_feature, example_label = train_data.__getitem__(final_train_ids[0])
    train_sampler = AGBatchSampler(batch_size, final_train_ids, True)
    train_data_loader = DataLoader(train_data, batch_sampler = train_sampler, num_workers = n_workers)
    
    ## 3. Load the embedding and ancestry group membership data for variants used for validation 
    valid_info_df = pd.read_csv(valid_info_file, sep = '\t', header = 0)
    valid_id_interval = np.cumsum(valid_info_df.n_var.values)
    if valid_subset_file != 'NA':
        valid_subset_ids = np.loadtxt(valid_subset_file, dtype = int)
        final_valid_ids = np.setdiff1d(valid_subset_ids, valid_exclude_ids)
    else:
        final_valid_ids = np.setdiff1d(np.arange(valid_id_interval[-1]), valid_exclude_ids)
    #
    valid_file_dict = process_h5_file('open', file_map = valid_info_df)
    valid_data = AGDataSet(valid_file_dict, valid_info_df, valid_id_interval, len(final_valid_ids))
    valid_sampler = AGBatchSampler(batch_size, final_valid_ids, False)
    valid_data_loader = DataLoader(valid_data, batch_sampler = valid_sampler, num_workers = n_workers)

    return train_file_dict, valid_file_dict, train_data_loader, valid_data_loader, example_feature.shape[0], example_label.shape[0]


## This function defines format for variant embedding and disease annotation data used for model fine-tuning 
class DISDataSet(Dataset):
    ## 0. Input arguments
        # feat_file (str): path of variant embedding h5 file 
        # label_file (str): path of variant disease annotation h5 file 

    def __init__(self, feat_file, label_file):
        super(DISDataSet, self).__init__()
        self.feat_h5 = h5py.File(feat_file, 'r')
        self.label_h5 = h5py.File(label_file, 'r')

    def __getitem__(self, index):
        feature = torch.tensor(np.abs(self.feat_h5['data'][index]), dtype = torch.float)
        label = torch.tensor(self.label_h5['data'][index], dtype = torch.float)
        return feature, label

    def __len__(self):
        return len(self.feat_h5['data'])


## This function computes the number of input/output neurons that represent each disease module in the VNN of fine-tuned DIS model  
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
def load_hierarchy_data(relation_file, layer_file, weight_file, root_input_size, min_m_size, max_m_size, weight_power):
    ## 0. Input arguments
        # relation_file (str): path of file that contains parent/children disease relationships  
        # layer_file (str): path of file that contains layer number of nodes
        # weight_file (str): path of file that contains weight info (proportion of positive/negative examples) of diseae terms
        # root_input_size (int): number of neurons in the last hidden layer of pre-trained model (i.e. input size for the root Diseae module/first layer of VNN model) 
        # min_m_size (int): mininum disease module size
        # max_m_size (int): maximum disease module size
        # weight_power (float): weight power (model hyperparameter) 

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

    ## 3. Compute the positive example weight for each disease module  
    weight_df = pd.read_csv(weight_file, sep = '\t')
    node_pos_ratio = weight_df.node_pos_weight.values
    pn_ratio = np.power(node_pos_ratio/(1 - node_pos_ratio), weight_power)
    module_pos_weight = pn_ratio/(pn_ratio+1)

    return parent_dict, root_id, input_module_size, output_module_size, module_pos_weight


## This function loads the variant embedding and ancestry group membership data for model pre-training
def load_finetune_data(train_pos_vep_file, train_pos_label_file, train_neg_vep_file, train_neg_label_file, valid_pos_vep_file, valid_pos_label_file, valid_neg_vep_file, valid_neg_label_file, batch_size = 128, n_workers = 4):
    ## 0. Input arguments
        # train_pos_vep_file (str): path of feature embedding h5 file for positive (pathogenic) variants used for training 
        # train_pos_label_file (str): path of disease annotation h5 file for positive variants used for training 
        # train_neg_vep_file (str): path of feature embedding h5 file for negative (neutral) variants used for training 
        # train_neg_label_file (str): path of disease annotation h5 file for negative variants used for training 
        # valid_pos_vep_file (str): path of feature embedding h5 file for positive variants used for validation
        # valid_pos_label_file (str): path of disease annotation h5 file for positive variants used for validation 
        # valid_neg_vep_file (str): path of feature embedding h5 file for negative variants used for validation
        # valid_neg_label_file (str): path of disease annotation h5 file for negative variants used for validation 
        # batch_size (int): batch size (Default: 128)
        # n_workers (int): number of workers (Default: 4)

    ## 1. Load the embedding and disease annotation data for variants used for training 
    train_pos_data = DISDataSet(train_pos_vep_file, train_pos_label_file)  
    N_train_pos = train_pos_data.__len__()
    train_neg_data = DISDataSet(train_neg_vep_file, train_neg_label_file)
    N_train_neg = train_neg_data.__len__()
    train_pos_bs = int(np.ceil(batch_size * N_train_pos/(N_train_pos + N_train_neg)))
    train_neg_bs = batch_size - train_pos_bs
    train_pos_data_loader = DataLoader(train_pos_data, batch_size = train_pos_bs, shuffle = True, num_workers = n_workers)
    train_neg_data_loader = DataLoader(train_neg_data, batch_size = train_neg_bs, shuffle = True, num_workers = n_workers)

    ## 2. Load the embedding and disease annotation data for variants used for annotation
    valid_pos_data = DISDataSet(valid_pos_vep_file, valid_pos_label_file)
    N_valid_pos = valid_pos_data.__len__()
    valid_neg_data = DISDataSet(valid_neg_vep_file, valid_neg_label_file)
    N_valid_neg = valid_neg_data.__len__()
    valid_pos_data_loader = DataLoader(valid_pos_data, batch_size = N_valid_pos, num_workers = n_workers)
    valid_neg_data_loader = DataLoader(valid_neg_data, batch_size = N_valid_neg, num_workers = n_workers)

    return train_pos_data_loader, train_neg_data_loader, valid_pos_data_loader, valid_neg_data_loader

