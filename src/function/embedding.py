## . /mnt/home/yhao1/miniforge3/envs/dis/bin/activate python
## Created by Yun Hao @FunctionLab 2025
## Code orginally from Kathy Chen's github repo sei-framework (https://github.com/FunctionLab/sei-framework) and selene (https://github.com/FunctionLab/selene)
## This script contains functions to generate variant embedding on levels of both transcriptional regulation (TR) and post-transcriptional regulation (PTR)


## Module
import os
import HTSeq
from docopt import docopt
from selene_sdk.sequences import Genome
from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run


## This function finds and replaces the <PATH> string in configuration file (part of https://github.com/FunctionLab/sei-framework/blob/main/1_variant_effect_prediction.py) 
def _finditem(obj, val):
    for k, v in obj.items():
        if hasattr(v, 'keywords'):
            _finditem(v.keywords, val)
        elif isinstance(v, dict):
            _finditem(v, val)
        elif isinstance(v, str) and '<PATH>' in v:
            obj[k] = v.replace('<PATH>', val)


## This function parses the chromosome size file
def parse_size_file(filename):
    use_chrs = set()
    chromlens = {}
    with open(filename, 'r') as file_handle:
        file_handle.readline()
        for line in file_handle:
            tok = line.split()
            chromlens[tok[0]] = int(tok[1])
            use_chrs.add(tok[0])
    # set default to zero
    genome_cov = HTSeq.GenomicArray(chromlens, stranded = True, typecode = "d")
    for interval, val in genome_cov.steps():
        genome_cov[interval] = 0
    # genome 2 genenames
    genome_geneid = HTSeq.GenomicArrayOfSets(chromlens, stranded = True)
    
    return use_chrs, chromlens, genome_cov, genome_geneid


## This function parses the bed file 
def parse_bed_file(filename, chromlens, genome_cov, genome_geneid):
    gene2strand = {}
    with open(filename, 'r') as file_handle:
        for line in file_handle:
            tok = line.split()
            chrom = tok[0]
            start = int(tok[1])
            end = int(tok[2])
            geneid = tok[3].split('.')[0]
            strand = '.'
            if len(tok) > 5:
                strand = tok[5]
            if chrom not in chromlens:
                continue
            interval = HTSeq.GenomicInterval(chrom, start, end, strand)
            genome_cov[interval] += 1
            genome_geneid[interval] += geneid
            gene2strand[geneid] = strand
    
    return gene2strand


## This function annotate the input vcf file with gene info for genic variants (within 500bp of genic region) 
def annotate_vcf_with_gene_info(input_file, output_file, hg_version, multi = True):
    ## 0. Input arguments 
        # input_file (str): path of input vcf file
        # output_file (str): path of output vcf file 
        # hg_version (str): version of human genome assembly: 'hg38' or 'hg19'
        # multi (bool): whether to skip variants mapped to genes on different strands (Default: True)

    ## 1. Process bed file based on the hg version 
    use_chrs, chromlens, genome_cov, genome_geneid = None, None, None, None
    gene2strand = None
    if hg_version == 'hg38':
        use_chrs, chromlens, genome_cov, genome_geneid = parse_size_file('resources/hg38.ensembl.genome')
        gene2strand = parse_bed_file('resources/gencode.v29.annotation.protein_coding_genes.hg38.bed', chromlens, genome_cov, genome_geneid)
    elif hg_version == 'hg19':
        use_chrs, chromlens, genome_cov, genome_geneid = parse_size_file('resources/hg19.ensembl.genome')
        gene2strand = parse_bed_file('resources/gencode.v28lift37.annotation.protein_coding_genes.bed', chromlens, genome_cov, genome_geneid)
    else:
        raise ValueError("--genome=<hg> must be 'hg19' or 'hg38'")
    
    ## 2. Annotate the gene info by row of input vcf file, write to output file  
    skip_chrom = ['Y', 'MT']
    output = None
    wchrom = None
    with open(input_file, 'r') as fin, \
            open(output_file, 'w+') as write_handle:
        for line in fin:
            if line.startswith('#'):
                continue
            tok = line.strip().split()
            chrom = tok[0]
            if "chr" in chrom:
                chrom = chrom[3:]
            start = int(tok[1]) - 1
            if len(tok) < 4:
                continue
            if len(tok) == 4:  # for now, assume missing ID col
                tok = tok[:2] + ['.'] + tok[2:]
            # remove multi-allelic SNPs
            toknew = []
            result = tok[0:5]
            if ',' in tok[4]:
                write_handle.write("{0}\t.\tNA\n".format('\t'.join(result)))
                continue
            if chrom not in use_chrs:
                write_handle.write("{0}\t.\tNA\n".format('\t'.join(result)))
                continue
            pos_position = HTSeq.GenomicPosition(chrom, start, '+')
            neg_position = HTSeq.GenomicPosition(chrom, start, '-')
            pos_cov = genome_cov[pos_position]
            neg_cov = genome_cov[neg_position]
            pos_cov_geneid = genome_geneid[pos_position]
            neg_cov_geneid = genome_geneid[neg_position]
            if pos_cov < 1 and neg_cov < 1:
                print('INFO: no strand info: {0}'.format(tok))
                write_handle.write("{0}\t.\tNA\n".format('\t'.join(result)))
                continue
            elif pos_cov > 0 and neg_cov > 0:
                if multi:
                    # print SNP twice with positive and negative strand
                    pos_result = result[:]
                    neg_result = result[:]
                    pos_result.append('+')
                    pos_result.append('|'.join(pos_cov_geneid))
                    neg_result.append('-')
                    neg_result.append('|'.join(neg_cov_geneid))
                    write_handle.write("{0}\n".format('\t'.join(pos_result)))
                    write_handle.write("{0}\n".format('\t'.join(neg_result)))
                    continue
                else:
                    print('INFO: --multi flag is off. '
                          'Skipping SNP mapped to genes on different strands '
                          'for SeqWeaver: {0}'.format(tok))
                    write_handle.write("{0}\t.\tNA\n".format('\t'.join(result)))
                    continue
            elif pos_cov > 0:
                result.append('+')
                result.append('|'.join(pos_cov_geneid))
            else:
                result.append('-')
                result.append('|'.join(neg_cov_geneid))
            write_handle.write("{0}\n".format('\t'.join(result)))
    
    return output_file

## This function computes TR or PTR variant embedding for input vcf file 
def compute_variant_embedding(vcf_file_loc, out_loc, hg_version, embed_type, device):
    ## 0. Input argument 
        # vcf_file_loc (str): path of input vcf file
        # out_loc (str): path of output file  
        # hg_version (str): version of human genome assembly: 'hg38' or 'hg19'
        # embed_type (str): type of embedding to be computed: 'Sei' for TR, 'Seqweaver' for PTR 
        # device (str): type of computing device: 'cpu' or 'cuda' (for gpu) 
   
    ## 1. Obtain the fasta file based on hg version  
    genome = Genome(os.path.join('.', 'resources', '{0}_UCSC.fa'.format(hg_version)))
    current_dir = os.path.abspath('')
    if device == 'cuda':
        cuda_bool = True
    else:
        cuda_bool = False

    ## 2. Specify the configuration file based of embedding type 
    # Transcriptional regulation model configuration file
    if embed_type == 'Sei':
        config_loc = './model/tr_dis/sei/sei_varianteffect_prediction.yml'
        out_folder = os.path.join(os.path.dirname(os.path.abspath(out_loc)), 'tr_embedding')
        os.makedirs(out_folder, exist_ok = True)
        annot_file = vcf_file_loc
    # Post-transcriptional regulation model configuration file
    if embed_type == 'Seqweaver':
        config_loc = './model/ptr_dis/seqweaver/human_seqweaver.yml'
        out_folder = os.path.join(os.path.dirname(os.path.abspath(out_loc)), 'ptr_embedding')
        os.makedirs(out_folder, exist_ok = True)
        # Annotate the vcf file with gene info
        annot_file = out_folder + '/' + os.path.basename(vcf_file_loc) + '_gene_annotated.vcf'
        annot_file = annotate_vcf_with_gene_info(vcf_file_loc, annot_file, hg_version) 

    ## 3. Read in and process the embedding configuration file
    configs = load_path(config_loc, instantiate = False)
    _finditem(configs, current_dir)
    configs["analyze_sequences"].bind(reference_sequence = genome, use_cuda = cuda_bool)
    configs["variant_effect_prediction"].update(vcf_files = [annot_file], output_dir = out_folder)
    
    ## 4. Parse configuration file, compute embedding 
    parse_configs_and_run(configs)    
    vep_file = out_folder + '/' + os.path.basename(annot_file)[0:-4] + '_diffs.h5'
    
    return vep_file 

