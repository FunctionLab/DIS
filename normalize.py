## . /mnt/home/yhao1/miniforge3/envs/dis/bin/activate python
## Created by Tess Marvin and Yun Hao @FunctionLab 2025
## This script normalize DIS+ predictions against neutral variant set 


## Module
import sys
import os
import argparse
sys.path.insert(0, 'src/function/')
import normalize


## 0. Inputs
parser = argparse.ArgumentParser(description = 'Z-normalize DIS+ predictions against neutral variants')
parser.add_argument('--neutral_file', type = str, default = None, help = 'neutral prediction .h5 file (or directory of .h5 files)')
parser.add_argument('--stats_file', type = str, default = None, help = 'precomputed neutral normalization stats .csv file')
parser.add_argument('--case_file', type = str, required = True, help = 'one case prediction .h5 file')
parser.add_argument('--out_dir', type = str, required = True, help = 'output directory')
parser.add_argument('--compare', action = 'store_true', help = 'output one CSV of per-node P-values and BH FDR')
parser.add_argument('--plot', action = 'store_true', help = 'output one case-versus-neutral plot per disease node')
parser.add_argument('--dataset_key', type = str, default = 'data', help = 'HDF5 dataset name (Default: data)')
args = parser.parse_args()
if not args.neutral_file and not args.stats_file:
    parser.error('provide --neutral_file or --stats_file')
if (args.compare or args.plot) and not args.neutral_file:
    parser.error('--compare and --plot require --neutral_file')
if not os.path.isfile(args.case_file):
    parser.error('--case_file must be a single existing .h5 file')
os.makedirs(os.path.dirname(args.out_dir), exist_ok = True)

## 1. Load or compute neutral normalization statistics
neutral_df = None
if args.stats_file:
    stats_df = normalize.load_norm_stats(args.stats_file)
    print('[OK] Loaded normalization stats -> ' + args.stats_file)
else:
    neutral_df = normalize.load_prediction_data(args.neutral_file, args.dataset_key)
    stats_df = normalize.compute_norm_stats(neutral_df)
    stats_file = args.out_dir + '_neutral_norm_stats.csv'
    stats_df.to_csv(stats_file, index = False)
    print('[OK] Computed normalization stats -> ' + stats_file)

## 2. Normalize all disease nodes and save HDF5
case_df = normalize.load_prediction_data(args.case_file, args.dataset_key)
case_norm_df = normalize.normalize_prediction_data(case_df, stats_df)
out_file = args.out_dir + '_normalized_predictions.h5'
normalize.save_normalized_predictions(case_norm_df, out_file, args.dataset_key)
print('[OK] Saved normalized predictions -> ' + out_file)

## 3. Optional comparisons and plots
if args.compare or args.plot:
    if neutral_df is None:
        neutral_df = normalize.load_prediction_data(args.neutral_file, args.dataset_key)
    neutral_norm_df = normalize.normalize_prediction_data(neutral_df, stats_df)

if args.compare:
    stat_df = normalize.compare_all_nodes(case_norm_df, neutral_norm_df)
    out_stats = args.out_dir + '_comparison_stats.csv'
    stat_df.to_csv(out_stats, index = False)
    print('[OK] Saved per-node comparisons -> ' + out_stats)

if args.plot:
    plot_dict = {'Neutral': neutral_norm_df, 'Case': case_norm_df}
    for node in case_norm_df.columns:
        out_plot = args.out_dir + '_plot_node_' + str(node) + '.png'
        normalize.plot_node_mean_ci95(plot_dict, node, 'Mean ± 95% CI (Node ' + str(node) + ')', out_plot)
    print('[OK] Saved ' + str(case_norm_df.shape[1]) + ' plots -> ' + args.out_dir + '_plot_node_')

