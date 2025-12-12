# Created by Tess Marvin and Yun Hao @FunctionLab 2025
# DeepLIFT explainer added (2025)

import sys
import os
import argparse
import h5py
import numpy as np
import pandas as pd
import torch
import random
sys.path.insert(0, 'src/function/')
import predict
import model  # uses model.DISModel as in predict.py

from captum.attr import DeepLift, DeepLiftShap

torch.set_num_threads(20)

parser = argparse.ArgumentParser(description="DeepLIFT explanations for DIS (TR/PTR) models")

# --- inputs (match your prediction script) ---
parser.add_argument('--vcf_file', type=str, default='NA')
parser.add_argument('--hg_version', type=str, default='NA')
parser.add_argument('--method', type=str, required=True, help="'Sei' for TR model, 'Seqweaver' for PTR model")
parser.add_argument('--out_name', type=str, required=True)
parser.add_argument('--vep_file', type=str)

# --- runtime ---
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--n_worker', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=-1)

# --- model config (defaults identical to your script) ---
parser.add_argument('--setting', type=int, default=1)
parser.add_argument('--ft_relation_file', type=str)
parser.add_argument('--ft_layer_file', type=str)
parser.add_argument('--ft_ag_info_file', type=str)
parser.add_argument('--ft_pretrain', type=bool, default=True)
parser.add_argument('--ft_min_module_size', type=int)
parser.add_argument('--ft_max_module_size', type=int)
parser.add_argument('--ft_n_unfreeze', type=int)
parser.add_argument('--ft_model', type=str)

# --- attribution settings ---
parser.add_argument('--attr_method', choices=['deeplift','deepliftshap'], default='deeplift')
parser.add_argument('--baseline', choices=['zero','dataset_mean','file'], default='zero')
parser.add_argument('--baseline_file', type=str, default=None, help="Path to .npy baseline vector if --baseline file")
parser.add_argument('--baseline_n', type=int, default=50000, help="N samples for dataset_mean or DLShap pool")
parser.add_argument('--targets', type=str, default='all', help="'all' or comma-separated integer indices")
parser.add_argument('--save_targets', type=str, default=None,
                    help="If set, compute attributions for all targets but only SAVE these "
                         "('all' or comma-separated integer indices). "
                         "If not set, falls back to --targets behavior.")
parser.add_argument('--attr_out', type=str, default=None, help="Optional path for the HDF5 out")

args = parser.parse_args()
# --- SET THE SEED EXPLICITLY ---
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- default paths like your main script ---
if args.setting == 1:
    if args.method == 'Sei':
        args.ft_relation_file = args.ft_relation_file or 'model/tr_dis/DO_tr_dis_node_parent.tsv'
        args.ft_layer_file    = args.ft_layer_file    or 'model/tr_dis/DO_tr_dis_node_layer.tsv'
        args.ft_ag_info_file  = args.ft_ag_info_file  or 'model/tr_dis/tr_dis_pretrain_info.txt'
        args.ft_min_module_size = args.ft_min_module_size or 10
        args.ft_max_module_size = args.ft_max_module_size or 100
        args.ft_n_unfreeze    = args.ft_n_unfreeze    or 0
        args.ft_model         = args.ft_model         or 'model/tr_dis/tr_dis_finetune.pt'
    elif args.method == 'Seqweaver':
        args.ft_relation_file = args.ft_relation_file or 'model/ptr_dis/DO_ptr_dis_node_parent.tsv'
        args.ft_layer_file    = args.ft_layer_file    or 'model/ptr_dis/DO_ptr_dis_node_layer.tsv'
        args.ft_ag_info_file  = args.ft_ag_info_file  or 'model/ptr_dis/ptr_dis_pretrain_info.txt'
        args.ft_min_module_size = args.ft_min_module_size or 10
        args.ft_max_module_size = args.ft_max_module_size or 100
        args.ft_n_unfreeze    = args.ft_n_unfreeze    or 0
        args.ft_model         = args.ft_model         or 'model/ptr_dis/ptr_dis_finetune.pt'
    else:
        raise ValueError("Unknown --method. Use 'Sei' or 'Seqweaver'.")

# --- embeddings (if VCF provided, rely on your embedding code path) ---
if args.vcf_file != 'NA' and args.hg_version != 'NA':
    import embedding
    args.vep_file = embedding.compute_variant_embedding(
        args.vcf_file, args.out_name, args.hg_version, args.method, args.device
    )
if not args.vep_file:
    raise ValueError("Provide --vep_file or (vcf_file + hg_version).")



# --- load models/config exactly like predict.py ---
ft_ag_model, pt_n_hidden_nodes = predict.load_ancestry_group_model(
    args.ft_ag_info_file, args.ft_pretrain, args.ft_n_unfreeze
)
ft_parent_dict, ft_root_id, ft_input_module_size, ft_output_module_size = predict.load_hierarchy_data(
    args.ft_relation_file, args.ft_layer_file, pt_n_hidden_nodes[-1],
    args.ft_min_module_size, args.ft_max_module_size
)

device = torch.device(args.device)
dis_model = model.DISModel(ft_ag_model, ft_parent_dict, ft_root_id, ft_input_module_size, ft_output_module_size)
ckpt = torch.load(args.ft_model, map_location='cpu')
state = ckpt.get('model_state_dict', ckpt)
dis_model.load_state_dict(state)
dis_model.to(device)
dis_model.eval()

# --- data loader (same helper as your script) ---
pred_loader = predict.load_predict_data(
    args.vep_file,
    start_id=args.start_id,
    end_id=args.end_id,
    b_size=args.batch_size,
    n_workers=args.n_worker
)

# determine input dim
first_batch = next(iter(pred_loader))
x0 = first_batch
in_dim = x0.shape[1]

# --- targets selection ---
T_total = len(ft_input_module_size)
all_target_indices = list(range(T_total))

def _parse_index_list(s):
    if s is None:
        return None
    s = s.strip().lower()
    if s == '' or s == 'all':
        return 'all'
    out = []
    for tok in s.split(','):
        t = int(tok)
        if t < 0 or t >= T_total:
            raise ValueError(f"Target {t} out of range [0, {T_total-1}]")
        out.append(t)
    return out

# Backward compatibility:
# - If --save_targets is provided, we use that for saving.
# - Else we use --targets (old behavior).
save_target_arg = args.save_targets if args.save_targets is not None else args.targets
save_target_indices = _parse_index_list(save_target_arg)
if save_target_indices == 'all':
    save_target_indices = list(range(T_total))
save_target_set = set(save_target_indices)

# --- baselines ---
def compute_dataset_mean_vec(loader, limit):
    total = 0
    running = None
    seen = 0
    with torch.no_grad():
        for batch in loader:
            xb = batch.to(device).float()
            if running is None:
                running = xb.sum(dim=0)
            else:
                running += xb.sum(dim=0)
            seen += xb.shape[0]
            if seen >= limit:
                break
    return (running / max(1, seen)).detach().cpu().numpy()

if args.baseline == 'zero':
    baseline_kind = 'zero'
    baseline_vec = None
elif args.baseline == 'dataset_mean':
    baseline_kind = 'dataset_mean'
    mean_vec = compute_dataset_mean_vec(pred_loader, args.baseline_n)
    baseline_vec = torch.tensor(mean_vec, dtype=torch.float32, device=device)
elif args.baseline == 'file':
    if not args.baseline_file or not os.path.exists(args.baseline_file):
        raise ValueError("--baseline file requires --baseline_file <.npy>")
    vec = np.load(args.baseline_file)
    if vec.shape[0] != in_dim:
        raise ValueError(f"Baseline vector dim {vec.shape[0]} != input dim {in_dim}")
    baseline_vec = torch.tensor(vec, dtype=torch.float32, device=device)
    baseline_kind = 'file'
else:
    raise ValueError("Unknown baseline")

def make_baseline(batch_like, method):
    B = batch_like.shape[0]
    if method == 'deeplift':
        # single baseline of shape (B, F)
        if baseline_kind == 'zero':
            return torch.zeros_like(batch_like)
        return baseline_vec.unsqueeze(0).expand(B, -1)
    else:
        # DeepLiftShap supports (m, F) or (B, m, F); we’ll use K fixed baselines
        K = min(25, B)
        if baseline_kind == 'zero':
            base = torch.zeros((K, in_dim), device=device, dtype=torch.float32)
        else:
            base = baseline_vec.unsqueeze(0).repeat(K, 1)
        return base

# --- captum explainer ---
explainer = DeepLift(dis_model) if args.attr_method == 'deeplift' else DeepLiftShap(dis_model)

# --- outputs ---
start_idx = pred_loader.dataset.start_index
end_idx = pred_loader.dataset.end_index
N = len(pred_loader.dataset)
default_attr_out = f"{args.out_name}_{args.method}_DEEPLIFT_{start_idx}_{end_idx}.h5"
attr_out_path = args.attr_out or default_attr_out
summary_csv_path = attr_out_path.replace('.h5', '_summary.csv')

h5f = h5py.File(attr_out_path, 'w')
h5f.attrs['method'] = args.attr_method
h5f.attrs['baseline'] = args.baseline
h5f.attrs['baseline_n'] = args.baseline_n
h5f.attrs['computed_targets'] = ','.join(map(str, all_target_indices))
h5f.attrs['saved_targets'] = ','.join(map(str, save_target_indices))
h5f.attrs['input_dim'] = in_dim
h5f.attrs['num_samples'] = N
h5f.attrs['model_path'] = args.ft_model
h5f.attrs['vep_file'] = args.vep_file

# Create datasets ONLY for the targets we intend to save
dsets = {
    t: h5f.create_dataset(f"attr_target_{t}", shape=(N, in_dim), dtype='float32', chunks=True)
    for t in save_target_indices
}

# --- attribution loop ---
offset = 0
for batch in pred_loader:
    xb = batch.to(device).float()
    xb.requires_grad_(True)

    base = make_baseline(xb, args.attr_method)

    # Compute for ALL targets but only write selected ones
    for t in all_target_indices:
        attr = explainer.attribute(inputs=xb, baselines=base, target=t)
        if t in save_target_set:
            dsets[t][offset:offset + xb.shape[0], :] = attr.detach().cpu().numpy().astype(np.float32)

    offset += xb.shape[0]

h5f.close()

# --- per-feature summary (mean and mean|.| across samples) ---
rows = []
with h5py.File(attr_out_path, 'r') as f:
    for t in save_target_indices:
        A = f[f"attr_target_{t}"]
        chunk = 8192
        mean = np.zeros(in_dim, dtype=np.float64)
        mean_abs = np.zeros(in_dim, dtype=np.float64)
        total = 0
        for i in range(0, N, chunk):
            sl = slice(i, min(N, i + chunk))
            a = A[sl]
            mean += a.sum(axis=0)
            mean_abs += np.abs(a).sum(axis=0)
            total += a.shape[0]
        mean /= max(1, total)
        mean_abs /= max(1, total)
        df = pd.DataFrame({
            'feature_index': np.arange(in_dim, dtype=int),
            'mean_attr': mean,
            'mean_abs_attr': mean_abs,
            'target': t
        })
        rows.append(df)

pd.concat(rows, ignore_index=True).to_csv(summary_csv_path, index=False)
print(f"[OK] DeepLIFT attributions computed for ALL {T_total} targets; saved {len(save_target_indices)} targets:\n"
      f"  • HDF5: {attr_out_path}\n"
      f"  • Summary: {summary_csv_path}")
