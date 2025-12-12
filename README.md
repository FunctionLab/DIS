# *DIS+: Interpretable Variant Pathogenicity Scores with Disease-specific Resolution*

**Yun Hao, Tess Marvin, Aviya Litman, Natalie Sauerwald, Christopher Y. Park, Denise G. O’Mahony, Vessela N. Kristensen, Olga G. Troyanskaya**

**Flatiron Institute, Princeton University**

Description of DIS+

DIS+ is the first disease-specific, AI-informed, and interpretable score for variant pathogenicity. DIS+ provides predictions for genome-wide regulatory variants across more than 100 diseases. DIS+ addresses a central unmet need in precision genomics: moving from disease-agnostic deleteriousness scores to disease-specific pathogenicity. With DIS+, instead of receiving a largely conservation-based assessment of a variant’s generic deleteriousness, researchers obtain a precise, disease-specific pathogenicity score with biochemical feature-level interpretation. 

Methodologically, DIS+ couples ancestry-informed pre-training with disease-ontology-guided fine-tuning in a visible AI framework reflecting the hierarchical relationship among diseases. Uniquely, DIS+ quantifies variant pathogenicity separately for transcriptional and post-transcriptional regulation, capturing disease-specific differences in the molecular mechanisms that drive pathogenesis. DIS+ also offers an interpretation module that computes feature-level attributions across thousands of regulatory features, including transcriptional factors and RNA-binding proteins. This design enables robust performance in data-scarce settings and reveals which regulatory programs drive disease risk. In addition to providing a disease-specific assessment and mechanistic interpretation, DIS+ significantly outperforms widely used, disease-agnostic predictors. More about DIS+ and training of the model are described in the following manuscript. 

## Setup 

### Requirements
Running DIS+ requires Python 3.10+ and Python packages PyTorch (>=2.2). Follow PyTorch installation steps [here](https://pytorch.org/). The other dependencies can be installed by running `pip install -r requirements.txt`.

### Install

Clone the repository then download and extract necessary resource files:
```bash
git clone https://github.com/FunctionLab/DIS.git
cd DIS
sh ./download_resources.sh
```

- [DIS+ model](https://doi.org/10.5281/zenodo.17902015)
- [DIS+ resource](https://doi.org/10.5281/zenodo.17902415)

## Usage

### Predicting DIS+ for regulatory variants
Command line ([example bash script](test/predict/test_var_predict.sh)):
```bash
python predict.py \ 
	--vcf_file <variant vcf file> \
	--hg_version <human genome assembly version> \ 
	--method <variant embedding generation method> \
	--out_name <output DIS+ prediction file>
```

Arguments:
- `--vcf_file`: input VCF file path ([example](test/predict/test_var.vcf))
- `--hg_version`: version of human genome assembly; `hg19` or `hg38`
- `--method`: method for generating variant embedding: `Sei` for transcriptional regulation embeddings (thus predicting disease impact on the transcriptional regulation level) or `Seqweaver` for post-transcriptional regulation embeddings (thus predicting disease impact on the post-transcriptional regulation level)
- `--out_name`: path or diretory for output DIS+ prediction files

Alternatively, if the user has the pre-computed variant embedding .h5 (VEP) file from running either Sei or Seqweaver, the following command line can be used: 
```bash
python predict.py \
	--vep_file <variant embedding file> \
	--method <variant embedding generation method> \
	--out_file <output DIS+ prediction file>
```
Arguments:
- `--vep_file`: input VEP file path ([example](test/test_seqweaver/ptr_embedding/test_var.vcf_gene_annotated_diffs.h5))

###  
```bash
python train.py \ 
	--mode <'pre-train'> \
	--out_name <output model files> \
	--pt_train_info_file <embedding-ancestry mapping file for training> \
	--pt_valid_info_file <embedding-ancestry mapping file for validation> \
	--pt_train_exclude_file <excluded variant file for training> \
	--pt_valid_exclude_file <excluded variant file for validation> \
	--pt_n_hidden <number of hidden neurons> \
	--pt_dr <dropout rate> \
	--pt_lr <learning rate> \
	--pt_l2 <L2 regularization factor> 
```
Arguments: 
- `--mode`: 'pre-train' for pre-training  
- `--out_name`: path or diretory for output files of pre-trained model 
- `--pt_train_info_file`: path for file containing embedding-ancestry group membership file map used for training ([example](test/pre_train/valid_file_map_10E-4.csv)) 
- `--pt_valid_info_file`: path for file containing embedding-ancestry group membership file map used for validation ([example](test/pre_train/valid_file_map_10E-4.csv))
- `--pt_train_exclude_file`: path for file containing the index of variants to be excluded from training ([example](test/pre_train/seqweaver_valid_exclusion_global.txt))
- `--pt_valid_exclude_file`: path for file containing the index of variants to be excluded from validation ([example](test/pre_train/seqweaver_valid_exclusion_global.txt))
- `--pt_n_hidden`: number of neurons in each hidden layer, seperated by comma (e.g. '1024,512,256')
- `--pt_dr`: dropout rate for pre-trained model
- `--pt_lr`: learning rate for pre-trained model 
- `--pt_l2`: L2 regularization factor of loss function for the pre-trained model

### Fine-tuning 
```bash
python train.py \
	--mode <'fine-tune'> \
	--out_name <output model files> \
	--ft_train_pos_vep_file <training variant embedding file> \
	--ft_train_pos_label_file <training positive variant disease annotation file> \
	--ft_train_neg_vep_file <training negative variant embedding file> \
	--ft_train_neg_label_file <training negative variant disease annotation file> \
	--ft_valid_pos_vep_file <validation variant embedding file> \
	--ft_valid_pos_label_file <validation positive variant disease annotation file> \
	--ft_valid_neg_vep_file <validation negative variant embedding file> \
	--ft_valid_neg_label_file <validation negative variant disease annotation file> \
	--ft_relation_file <disease relationship file> \
	--ft_layer_file <disease layer number file> \
	--ft_weight_file <disease term weight file> \
	--ft_weight_pwr <weight power> \
	--ft_ag_info_file <pre-trained model configuration file> \
	--ft_min_module_size <mininum disease module size> \
	--ft_max_module_size <maxinum disease module size> \
	--ft_n_unfreeze <number of pre-trained layers to unfreeze> \
	--ft_lr <learning rate> \
	--ft_l2 <L2 regularization factor> \
	--ft_mrl_margin <margin factor of margin rank loss> \
	--ft_mrl_coeff <coefficient of margin rank loss>
``` 
Arguments:
- `--mode`: 'fine-tune' for fine-tuning
- `--out_name`: path or diretory for output files of fine-tuned model
- `--ft_train_pos_vep_file`: path for positive variant embedding h5 file used for training ([example](test/test_seqweaver/ptr_embedding/test_var.vcf_gene_annotated_diffs.h5))
- `--ft_train_pos_label_file`: path for positive variant disease annotation h5 file used for training ([example](test/fine_tune/DO_seqweaver_HGMD_valid_var_node_map.h5))
- `--ft_train_neg_vep_file`: path for negative variant embedding h5 file used for training ([example](test/test_seqweaver/ptr_embedding/test_var.vcf_gene_annotated_diffs.h5))
- `--ft_train_neg_label_file`: path for negative variant disease annotation h5 file used for training ([example](test/fine_tune/DO_seqweaver_NEUTRAL_valid_var_node_map_5kb.h5)) 
- `--ft_valid_pos_vep_file`: path for positive variant embedding h5 file used for validation ([example](test/test_seqweaver/ptr_embedding/test_var.vcf_gene_annotated_diffs.h5))
- `--ft_valid_pos_label_file`: path for positive variant disease annotation h5 file used for validation ([example](test/fine_tune/DO_seqweaver_HGMD_valid_var_node_map.h5))
- `--ft_valid_neg_vep_file`: path for negative variant embedding h5 file used for validation ([example](test/test_seqweaver/ptr_embedding/test_var.vcf_gene_annotated_diffs.h5))
- `--ft_valid_neg_label_file`: path for negative variant disease annotation h5 file used for validation ([example](test/fine_tune/DO_seqweaver_NEUTRAL_valid_var_node_map_5kb.h5))
- `--ft_relation_file`: path for parent/children disease relationship file ([example](test/fine_tune/DO_seqweaver_noncoding_parsed_250_node_parent.tsv))
- `--ft_layer_file`: path for disease layer number file ([example](test/fine_tune/DO_seqweaver_noncoding_parsed_250_node_layer.tsv))
- `--ft_weight_file`: path for disease term weight file ([example](test/fine_tune/DO_seqweaver_noncoding_parsed_250_node_weight.tsv))
- `--ft_weight_pwr`: weight power
- `--ft_ag_info_file`: path for pre-trained model configuration file ([example](dis_finetune_mr_seqweaver_noncoding_pretrain_model_info_1.txt))
- `--ft_min_module_size`: mininum disease module size for fine-tuned model
- `--ft_max_module_size`: maxinum disease module size for fine-tuned model
- `--ft_n_unfreeze`: number of pre-trained layers to unfreeze for fine-tuning
- `--ft_lr`: learning rate for fine-tuned model
- `--ft_l2`: L2 regularization factor of loss function for the fine-tuned model
- `--ft_mrl_margin`: margin factor of margin rank loss function for the fine-tuned model
- `--ft_mrl_coeff`: coefficient of margin rank loss function for the fine-tuned model

### Model interpretation
DIS+ includes a built-in interpretability module to provide mechanistic interpretation of model predictions. The script **`explain.py`** computes per-feature attributions for each variant embedding using **DeepLIFT** or **DeepLIFT-SHAP** (via Captum). These explanations allow users to understand which regulatory features drive disease-specific pathogenicity predictions for both transcriptional and post-transcriptional models.

### Running the Interpretability Module
```bash
python explain.py \
	--method <variant embedding generation method> \
	--vcf_file <variant vcf file> \
	--hg_version <human genome assembly version> \
	--out_name <output DIS+ interpretation file prefix> \
	--attr_method <attritbuion computing method> \
	--baseline <attribution baseline reference>
```
Arguments:
- `--vcf_file`: input VCF file path
- `--hg_version`: version of human genome assembly; `hg19` or `hg38`
- `--method`: method for generating variant embedding: `Sei` for transcriptional regulation embeddings (thus predicting disease impact on the transcriptional regulation level) or `Seqweaver` for post-transcriptional regulation embeddings (thus predicting disease impact on the post-transcriptional regulation level)
- `--out_name`: path or diretory for output DIS+ interpretation files
- `--attr_method`: method for computing the feature attribution scores: `deeplift` (default) or `deepliftshap` (uses multi-baseline sampling) 
- `--baseline`: baseline reference used for DeepLIFT attribution: `zero` (default), `dataset_mean`, or `file` (need to specify with `--baseline_file`)

Optional Arguments:
- `--baseline_file` (required if `--baseline file`): path to `.npy` file containing the baseline vector (vector dimension must match embedding size)
- `--baseline_n` (if if `--baseline dataset_mean`): number of samples used for computing dataset mean or SHAP baseline pool (default: 50000)
- `targets`: string containing the disease node IDs of which attribution scores will be computed, seperated by comma, or `all` (default) will compute for every disease of DIS+ 
- `save_targets`: string containing the disease node IDs of which attribution scores will NOT be computed, or `None` (default) will compute for every disease of DIS+
- `attr_out`: custom path for the output interpretation file, if user wants to output file in a different location than default, which follows the `--out_name` (`<out_name>_<method>_DEEPLIFT_*.h5`) 

Alternatively, if the user has the pre-computed variant embedding .h5 (VEP) file from running either Sei or Seqweaver, the following command line can be used:
```bash
python explain.py \
	--method <variant embedding generation method> \
	--vep_file <variant embedding file> \
	--out_name <output DIS+ interpretation file prefix> \
	--attr_method <attritbuion computing method> \
	--baseline <attribution baseline reference>
``` 
---
#### Notes
Large numbers of targets or variants produce large HDF5 files; use `--save_targets` to limit output size.

Feature interpretation works for both transcriptional (Sei) and post-transcriptional (Seqweaver) embeddings.

The explainer loads the same fine-tuned DIS+ model used in prediction.

## Help
Please post in the Github issues or e-mail Yun Hao [yhao@flatironinstitute.org](mailto:yhao@flatironinstitute.org) with any questions about the repository, requests for more data, etc.
