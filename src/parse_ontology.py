# . /mnt/home/yhao1/miniforge3/envs/dis/bin/activate python
## created by Yun Hao @FunctionLab 2023
## This script processes Disease Ontology term relationship file to generate sorted hierarchy files for constructing the visible neural network (VNN) architecture


## Module
import argparse
import numpy as np
import pandas as pd
from itertools import chain

## This function finds all the ancestor terms of given disease term based on parent-child relationships
def find_all_ancestors(query_term, relation_df):
    ## 0. Input arguments
        # query_term (str): given disease term ID
        # relation_df (pandas DataFrame): data frame that contains parent-child disease term relationships
    
    query_parent = [[query_term]]
    parent_id = relation_df.child.isin([query_term])
    while sum(parent_id) > 0:
        parent_terms = relation_df[parent_id].parent.values.tolist()
        query_parent.append(parent_terms)
        parent_id = relation_df.child.isin(parent_terms)
    unique_parents = np.unique(list(chain(*query_parent))).tolist()
    
    return unique_parents


## 0. Inputs 
parser = argparse.ArgumentParser(description = 'Process some integers.')
parser.add_argument('--relation_file', action = 'store', dest = 'relation_file', type = str)
parser.add_argument('--term_map_file', action = 'store', dest = 'term_map_file', type = str)
parser.add_argument('--var_list_file', action = 'store', dest = 'var_list_file', type = str, default = 'NA')
parser.add_argument('--min_var', action = 'store', dest = 'min_var', type = int)
parser.add_argument('--min_train_var', action = 'store', dest = 'min_train_var', type = int, default = 0)
parser.add_argument('--min_valid_var', action = 'store', dest = 'min_valid_var', type = int, default = 0)
parser.add_argument('--out_name', action = 'store', dest = 'out_name', type = str)
args = parser.parse_args()

## 1. Read in input files
# Read in disease term parent-child relationship file
rela_df = pd.read_csv(args.relation_file, sep = '\t', header = 0)
all_terms = np.union1d(rela_df.parent.values, rela_df.child.values)
# Read in variant-disease term mapping file 
term_map_df = pd.read_csv(args.term_map_file)
# Filter mapping file by input variant list if any
if args.var_list_file != 'NA':
    all_vars = np.loadtxt(args.var_list_file, dtype = str)
    term_map_df = term_map_df[term_map_df.identifier.isin(all_vars)] 

## 2. Map each variant to all the ancestor terms of HGMD-curated term
# Find all ancestors for each disease term 
all_map_terms = [tdv.split(',') for tdv in term_map_df.DOID.values]
all_unique_terms = np.unique(list(chain(*all_map_terms)))
term_ancestor = {}
for aut in all_unique_terms:
    term_ancestor[aut] = find_all_ancestors(aut, rela_df)
# Map each variant to all the ancestor terms of HGMD-curated term
term_ancestor_list = []
for itr, tmd_row in term_map_df.iterrows():
    tmd_dt = tmd_row['DOID'].split(',')
    tmd_anc = [term_ancestor[td] for td in tmd_dt]
    tmd_uni_anc = np.unique(list(chain(*tmd_anc)))
    tua_df = pd.DataFrame({'identifier': tmd_row['identifier'], 'DOID': tmd_uni_anc})
    term_ancestor_list.append(tua_df)
term_ancestor_df = pd.concat(term_ancestor_list)
term_ancestor_df['chrom'] = [tiv.split('_')[0] for tiv in term_ancestor_df.identifier.values]
# Remove disease terms with too few overall mapped variants 
term_counts = term_ancestor_df.DOID.value_counts()
filter_terms = term_counts[term_counts > args.min_var].index
# Remove disease terms with too few mapped variants in training set
train_chroms = ['chr2', 'chr1', 'chrX', 'chr17', 'chr11', 'chr16', 'chr3', 'chr19', 'chr12', 'chr7','chr4', 'chr18', 'chr21', 'chr20']
train_term_ancestor_df = term_ancestor_df[term_ancestor_df.chrom.isin(train_chroms)]
train_term_counts = train_term_ancestor_df.DOID.value_counts()
train_filter_terms = train_term_counts[train_term_counts > args.min_train_var].index
filter_terms = np.intersect1d(filter_terms, train_filter_terms)
# Remove disease terms with too few mapped variants in validation set
valid_chroms = ['chr6', 'chr15', 'chr22']
valid_term_ancestor_df = term_ancestor_df[term_ancestor_df.chrom.isin(valid_chroms)]
valid_term_counts = valid_term_ancestor_df.DOID.value_counts()
valid_filter_terms = valid_term_counts[valid_term_counts > args.min_valid_var].index
filter_terms = np.intersect1d(filter_terms, valid_filter_terms)

## 2. Sort all disease terms by layers based on parent-child relationships 
# Filter relationship data frame by remaining disease terms 
rela_df1 = rela_df[rela_df.parent.isin(filter_terms)]
rela_df2 = rela_df1[rela_df1.child.isin(filter_terms)]
rela_df2 = rela_df2.drop_duplicates()
root_terms = np.setdiff1d(rela_df2.parent.values, rela_df2.child.values)
lowest_terms = np.setdiff1d(rela_df2.child.values, rela_df2.parent.values)
upper_terms = np.setdiff1d(filter_terms, lowest_terms)
# Define layer of all disease terms, start with layer 0 of root terms  
start_terms = root_terms
current_upper_terms = np.intersect1d(start_terms, upper_terms)
current_layer = 0
layer_dict = {}
parent_dict = {}
for rt in root_terms:
    layer_dict[rt] = current_layer
# Extend to upper layers via parent-child relationships
while len(current_upper_terms) > 0:
    current_layer = current_layer + 1
    start_id = rela_df2.parent.isin(start_terms)
    child_terms = rela_df2[start_id].child.values
    parent_terms = rela_df2[start_id].parent.values
    for ct in child_terms:
       layer_dict[ct] = current_layer
    # Store parent terms of each disease 
    for ict, ct in enumerate(child_terms):
        if ct in parent_dict.keys():
            if parent_terms[ict] in parent_dict[ct]:
                continue
            else:
                parent_dict[ct].append(parent_terms[ict]) 
        else:
            parent_dict[ct] = [parent_terms[ict]]
    # Start next query with new children terms  
    start_terms = child_terms
    current_upper_terms = np.intersect1d(start_terms, upper_terms)
    

## 3. Generate sorted disease ontology hierarchy files 
# Generate layer number and disease node number file 
layer_df = pd.DataFrame.from_dict(layer_dict, orient = 'index')
layer_df.columns = ['layer_number']
layer_df['node_name'] = layer_df.index
layer_df = layer_df.sort_values(by = 'layer_number')
layer_df['node_id'] = np.arange(layer_df.shape[0])
out_node_df = layer_df.loc[:, ['node_id', 'node_name']]
out_node_df.to_csv(args.out_name + '_' + str(args.min_var) + '_node_id.tsv', sep = '\t', index = False)
out_layer_df = layer_df.loc[:, ['node_id', 'layer_number']]
out_layer_df.to_csv(args.out_name + '_' + str(args.min_var) + '_node_layer.tsv', sep = '\t', index = False)
# Generate variant-disease node mapping file  
term_ancestor_df1 = pd.merge(term_ancestor_df, out_node_df, left_on = 'DOID', right_on = 'node_name')
out_map_df = term_ancestor_df1.loc[:, ['identifier', 'node_id']].sort_values(by = 'identifier')
out_map_df.to_csv(args.out_name + '_' + str(args.min_var) + '_node_var_map.tsv', sep = '\t', index = False)
# Generate disease node weight file 
out_map_count = out_map_df.node_id.value_counts()
out_node_weight = 1 - out_map_count/(2*out_map_count.max())
out_weight_df = pd.DataFrame({'node_id': out_map_count.index, 'node_pos_weight': out_node_weight})
out_weight_df = out_weight_df.reset_index(drop = True)
out_weight_df = out_weight_df.sort_values(by = 'node_id')
out_weight_df.to_csv(args.out_name + '_' + str(args.min_var) + '_node_weight.tsv', sep = '\t', index = False)
# Generate parent node mapping file 
child_vec = []
parent_vec = []
for key,value in parent_dict.items():
    key_id = out_node_df[out_node_df.node_name == key].node_id.values[0]
    child_vec.append(key_id)
    parent_id = out_node_df[out_node_df.node_name.isin(value)].node_id.values
    parent_vec.append(','.join([str(pi) for pi in parent_id]))
out_rela_df = pd.DataFrame({'child_node_id': child_vec, 'parent_node_id': parent_vec})
out_rela_df = out_rela_df.sort_values(by = 'parent_node_id')
out_rela_df.to_csv(args.out_name + '_' + str(args.min_var) + '_node_parent.tsv', sep = '\t', index = False)

