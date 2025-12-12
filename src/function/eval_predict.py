# predict.py
# Updated by Tess Marvin & Yun Hao @FunctionLab 2024

import sys
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
sys.path.insert(0, 'src/function/')
import model

class DISPredictSet(Dataset):
    def __init__(self, feat_file, label_file=None, start_index=0, end_index=-1):
        super(DISPredictSet, self).__init__()
        self.feat_h5 = h5py.File(feat_file, 'r')
        self.label_h5 = h5py.File(label_file, 'r') if label_file else None
        self.start_index = start_index
        self.end_index = end_index if end_index >= 0 else len(self.feat_h5['data'])

    def __getitem__(self, index):
        feature = torch.tensor(np.abs(self.feat_h5['data'][index + self.start_index]), dtype=torch.float)
        if self.label_h5:
            label = torch.tensor(self.label_h5['data'][index + self.start_index], dtype=torch.float)
            return feature, label
        return feature

    def __len__(self):
        return self.end_index - self.start_index

def load_predict_data(vep_file, label_file=None, start_id=0, end_id=-1, b_size=128, n_workers=4):
    predict_data = DISPredictSet(vep_file, label_file, start_id, end_id)
    predict_data_loader = DataLoader(predict_data, batch_size=b_size, num_workers=n_workers)
    return predict_data_loader

def compute_disease_module_stat(layer_df, parent_dict, root_id, root_input_size, min_m_size, max_m_size):
    n_max_layer = layer_df.layer_number.max()
    layer_n = layer_df.layer_number.values
    output_size = max_m_size - layer_n / n_max_layer * (max_m_size - min_m_size)
    output_size = np.ceil(output_size).astype(int)

    input_size = np.zeros(layer_df.shape[0], dtype=int)
    for lnv in range(layer_df.shape[0]):
        if lnv in root_id:
            input_size[lnv] = root_input_size
        else:
            input_size[lnv] = np.sum(output_size[parent_dict[lnv]])

    return input_size, output_size

def load_hierarchy_data(relation_file, layer_file, root_input_size, min_m_size, max_m_size):
    relation_df = pd.read_csv(relation_file, sep='\t')
    parent_dict = {}
    for i in range(relation_df.shape[0]):
        i_child = relation_df.child_node_id.values[i]
        i_parent = relation_df.parent_node_id.values[i].split(',')
        parent_dict[i_child] = np.array(i_parent).astype(int)

    layer_df = pd.read_csv(layer_file, sep='\t')
    root_id = layer_df[layer_df.layer_number == 0].node_id.values
    input_module_size, output_module_size = compute_disease_module_stat(
        layer_df, parent_dict, root_id, root_input_size, min_m_size, max_m_size)

    return parent_dict, root_id, input_module_size, output_module_size

def load_ancestry_group_model(ag_info_file, pre_train=False, n_unfreeze=0):
    with open(ag_info_file, 'r') as info_f:
        ag_info_dict = dict(line.strip().split(': ') for line in info_f.readlines())

    n_input_feat = int(ag_info_dict['Number of input features'])
    n_output_feat = int(ag_info_dict['Number of output features'])
    n_hidden_nodes = np.array(ag_info_dict['Number of hidden neurons'].split(','), dtype=int)
    dr = float(ag_info_dict['Dropout rate'])
    ag_model = model.AncestryGroupModel(n_input_feat, n_hidden_nodes, n_output_feat, dr)

    if pre_train:
        model_state = torch.load(ag_info_dict['Optimal pretrain model'])
        ag_model.load_state_dict(model_state['model_state_dict'])
        n_freeze = len(n_hidden_nodes) - n_unfreeze
        for nf in range(n_freeze):
            for param in ag_model.hidden[nf].parameters():
                param.requires_grad = False

    return ag_model, n_hidden_nodes

def predict_disease_impact(pred_data_loader, ag_model, dis_parent_dict, dis_root, dis_in_size, dis_out_size,
                            model_name, model_device='cpu', num_repeat=20, return_outputs=False):
    dis_model = model.DISModel(ag_model, dis_parent_dict, dis_root, dis_in_size, dis_out_size)
    model_state = torch.load(model_name)
    dis_model.load_state_dict(model_state['model_state_dict'])
    dis_model.to(model_device)
    dis_model.eval()

    all_preds = []
    all_labels = []

    for batch in pred_data_loader:
        if isinstance(batch, (list, tuple)):
            batch_feature, batch_label = batch
            all_labels.append(batch_label)
        else:
            batch_feature = batch

        batch_feature = batch_feature.to(model_device)
        batch_pred_rep = torch.zeros(num_repeat, batch_feature.shape[0], len(dis_in_size), device=model_device)

        for i in range(num_repeat):
            batch_pred_rep[i] = dis_model(batch_feature)

        batch_pred = torch.mean(batch_pred_rep, dim=0)
        all_preds.append(batch_pred.detach().cpu())

    y_preds = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy() if all_labels else None

    return (y_preds, y_true) if return_outputs else 1
