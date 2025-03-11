import json
import torch
import numpy as np
from tqdm import tqdm, trange
from Bio import SeqIO
from pathlib import Path
import torch.nn.functional as F



def get_index_protein_dic(protein_list):
    return {index: protein for index, protein in enumerate(protein_list)}

def make_parent_dir(path):
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

def dot_product(z1, z2):
    return torch.sigmoid(torch.mm(z1, z2.t()))

def get_search_list(search_result):
    search_list = []
    with open(search_result) as fp:
        for line in tqdm(fp, desc='Get search list'):
            line_list = line.strip().split('\t')
            protein1 = line_list[0].split('.pdb')[0]
            protein2 = line_list[1].split('.pdb')[0]
            score = eval(line_list[2])
            search_list.append(((protein1, protein2), score))
    return search_list

def cos_similarity(z1, z2):
    eps = 1e-8
    z1_n, z2_n = z1.norm(dim=1)[:, None], z2.norm(dim=1)[:, None]
    z1_norm = z1 / torch.max(z1_n, eps * torch.ones_like(z1_n))
    z2_norm = z2 / torch.max(z2_n, eps * torch.ones_like(z2_n))
    sim_mt = torch.mm(z1_norm, z2_norm.transpose(0, 1))
    return sim_mt

def euclidean_similarity(z1, z2):
    eps = 1
    dist_matrix = torch.cdist(z1, z2)
    sim_matrix = 1 / (dist_matrix + eps)
    return sim_matrix

def tensor_to_list(tensor):
    decimals = 4
    numpy_array = tensor.cpu().numpy()
    return np.round(numpy_array, decimals=decimals).tolist()

def read_fasta(fn_fasta):
    prot2seq = {}
    with open(fn_fasta) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            prot = record.id
            prot2seq[prot] = seq
    return list(prot2seq.keys()), prot2seq