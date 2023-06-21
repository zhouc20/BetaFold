import numpy as np
import glob
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple
from torch_sparse import SparseTensor
from torch_geometric.data import Data as PygData
import os
from tqdm import tqdm
import pandas as pd
import pickle
from .torsion_angle_tool import side_chain_torsion_angles


def one_hot_acid(x):
    acid_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z', 'X', '*', '-']
    return acid_list.index(x)


def load_pyg_data_old(max_length=512):
    """
    FASTA格式与氨基酸的对应关系：
    A  alanine               P  proline
    B  aspartate/asparagine  Q  glutamine
    C  cystine               R  arginine
    D  aspartate             S  serine
    E  glutamate             T  threonine
    F  phenylalanine         U  selenocysteine
    G  glycine               V  valine
    H  histidine             W  tryptophan
    I  isoleucine            Y  tyrosine
    K  lysine                Z  glutamate/glutamine
    L  leucine               X  any
    M  methionine            *  translation stop
    N  asparagine            -  gap of indeterminate length
    """
    files = glob.glob('ProTstab2EachSpeciesDatasets/*.csv')
    data_list_train = []
    data_list_test = []
    data_list_valid = []
    for file in files:
        train_path = file.split('.')[0] + "_train.txt"
        test_path = file.split('.')[0] + "_test.txt"
        valid_path = file.split('.')[0] + "_valid.txt"
        # with open(train_path, "r") as f:
        #     train_idx = np.array(f.read()).astype(int).tolist()
        #     f.close()
        # with open(valid_path, "r") as f:
        #     valid_idx = np.array(f.read()).astype(int).tolist()
        #     f.close()
        # with open(test_path, "r") as f:
        #     test_idx = np.array(f.read()).astype(int).tolist()
        #     f.close()
        # train_idx = np.array(pd.read_csv(train_path)).astype(int).tolist()
        # test_idx = np.array(pd.read_csv(test_path)).astype(int).tolist()
        # valid_idx = np.array(pd.read_csv(valid_path)).astype(int).tolist()
        train_idx = np.loadtxt(train_path, dtype=int)
        valid_idx = np.loadtxt(valid_path, dtype=int)
        test_idx = np.loadtxt(test_path, dtype=int)
        # print(train_idx)
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                else:
                    idx, name, x, species, y = row
                    idx = int(idx)
                    x = torch.tensor(list(map(one_hot_acid, x)), dtype=torch.long)
                    # x = F.one_hot(x, num_classes=26)
                    if x.shape[0] > max_length:
                        x = x[:max_length]
                    edge_index_0 = torch.arange(0, x.shape[0], dtype=torch.long).unsqueeze(1).repeat(1, 2).reshape(-1)
                    edge_index_1 = torch.arange(0, x.shape[0], dtype=torch.long).unsqueeze(1).repeat(1, 2).reshape(-1)
                    edge_index_1[torch.arange(0, x.shape[0] * 2, 2)] -= 1
                    edge_index_1[torch.arange(1, x.shape[0] * 2, 2)] += 1
                    edge_index = torch.cat([edge_index_0[1: -1].unsqueeze(0), edge_index_1[1: -1].unsqueeze(0)], dim=0)
                    edge_attr = torch.ones([edge_index.shape[1]], dtype=torch.long)
                    data = PygData(x=x, edge_index=edge_index, edge_attr=edge_attr, idx=torch.tensor([idx], dtype=torch.long),  #, name=[name], species=[species],
                                   y=torch.tensor([float(y)], dtype=torch.float), len_seq=torch.tensor((x.shape[0]), dtype=torch.long))
                    # print(idx)
                    if idx in train_idx:
                        data_list_train.append(data)
                    elif idx in test_idx:
                        data_list_test.append(data)
                    elif idx in valid_idx:
                        data_list_valid.append(data)
                    else:
                        raise NotImplementedError
    return data_list_train, data_list_valid, data_list_test


def load_pyg_data(max_length=512):
    """
    FASTA格式与氨基酸的对应关系：
    A  alanine               P  proline
    B  aspartate/asparagine  Q  glutamine
    C  cystine               R  arginine
    D  aspartate             S  serine
    E  glutamate             T  threonine
    F  phenylalanine         U  selenocysteine
    G  glycine               V  valine
    H  histidine             W  tryptophan
    I  isoleucine            Y  tyrosine
    K  lysine                Z  glutamate/glutamine
    L  leucine               X  any
    M  methionine            *  translation stop
    N  asparagine            -  gap of indeterminate length
    """

    data_list_train = []
    data_list_test = []
    train_path = 'ProTstab2_dataset_new/train_dataset.csv'
    test_path = 'ProTstab2_dataset_new/test2_dataset.csv'
    with open(train_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            else:
                idx, name, x, species, y = row
                idx = int(idx)
                x = torch.tensor(list(map(one_hot_acid, x)), dtype=torch.long)
                # x = F.one_hot(x, num_classes=26)
                if x.shape[0] > max_length:
                    x = x[:max_length]
                edge_index_0 = torch.arange(0, x.shape[0], dtype=torch.long).unsqueeze(1).repeat(1, 2).reshape(-1)
                edge_index_1 = torch.arange(0, x.shape[0], dtype=torch.long).unsqueeze(1).repeat(1, 2).reshape(-1)
                edge_index_1[torch.arange(0, x.shape[0] * 2, 2)] -= 1
                edge_index_1[torch.arange(1, x.shape[0] * 2, 2)] += 1
                edge_index = torch.cat([edge_index_0[1: -1].unsqueeze(0), edge_index_1[1: -1].unsqueeze(0)], dim=0)
                edge_attr = torch.ones([edge_index.shape[1]], dtype=torch.long)
                data = PygData(x=x, edge_index=edge_index, edge_attr=edge_attr, idx=torch.tensor([idx], dtype=torch.long),  #, name=[name], species=[species],
                               y=torch.tensor([float(y)], dtype=torch.float), len_seq=torch.tensor((x.shape[0]), dtype=torch.long))
                data_list_train.append(data)

    with open(test_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            else:
                idx, name, x, species, y = row
                idx = int(idx)
                x = torch.tensor(list(map(one_hot_acid, x)), dtype=torch.long)
                # x = F.one_hot(x, num_classes=26)
                if x.shape[0] > max_length:
                    x = x[:max_length]
                edge_index_0 = torch.arange(0, x.shape[0], dtype=torch.long).unsqueeze(1).repeat(1, 2).reshape(-1)
                edge_index_1 = torch.arange(0, x.shape[0], dtype=torch.long).unsqueeze(1).repeat(1, 2).reshape(-1)
                edge_index_1[torch.arange(0, x.shape[0] * 2, 2)] -= 1
                edge_index_1[torch.arange(1, x.shape[0] * 2, 2)] += 1
                edge_index = torch.cat([edge_index_0[1: -1].unsqueeze(0), edge_index_1[1: -1].unsqueeze(0)], dim=0)
                edge_attr = torch.ones([edge_index.shape[1]], dtype=torch.long)
                data = PygData(x=x, edge_index=edge_index, edge_attr=edge_attr, idx=torch.tensor([idx], dtype=torch.long),  #, name=[name], species=[species],
                               y=torch.tensor([float(y)], dtype=torch.float), len_seq=torch.tensor((x.shape[0]), dtype=torch.long))
                data_list_test.append(data)

    return data_list_train, data_list_test


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_pygdata_3d(path):
    '''
    train_path = 'StructuredDatasets/train_dataset.pkl'
    test_path = 'StructuredDatasets/test2_dataset.pkl'
    '''
    data = pickle_load(path)
    Tms = [float(v['Tm']) for v in data.values()]
    sequences = [v['simple_fasta'] for v in data.values()]
    structures = [v['structure'] for v in data.values()]
    del data  # free memory
    atom_names = [v['atom_name'] for v in structures]
    xs = [v['x'] for v in structures]
    ys = [v['y'] for v in structures]
    zs = [v['z'] for v in structures]
    residue_names = [v['residue_name'] for v in structures]
    data_list = []
    
    with tqdm(desc=f'preprocess...', total=len(Tms), unit='proteins') as pbar:
        for i in range(len(Tms)):
            seq = torch.tensor(list(map(one_hot_acid, sequences[i])), dtype=torch.long)
            js = [k for k, name in enumerate(atom_names[i]) if name == 'CA']
            x = torch.tensor([xs[i][j] for j in js], dtype=torch.float32)
            y = torch.tensor([ys[i][j] for j in js], dtype=torch.float32)
            z = torch.tensor([zs[i][j] for j in js], dtype=torch.float32)
            pos = torch.stack([x, y, z], dim=-1)
            edge_index_0 = torch.arange(0, seq.shape[0], dtype=torch.long).unsqueeze(1).repeat(1, 2).reshape(-1)
            edge_index_1 = torch.arange(0, seq.shape[0], dtype=torch.long).unsqueeze(1).repeat(1, 2).reshape(-1)
            edge_index_1[torch.arange(0, seq.shape[0] * 2, 2)] -= 1
            edge_index_1[torch.arange(1, seq.shape[0] * 2, 2)] += 1
            edge_index = torch.cat([edge_index_0[1: -1].unsqueeze(0), edge_index_1[1: -1].unsqueeze(0)], dim=0)
            edge_attr = torch.ones([edge_index.shape[1]], dtype=torch.long)
            torsion_angle = side_chain_torsion_angles(sequences[i], structures[i], pad=-1.0)
            data_3d = PygData(x=seq, edge_index=edge_index, edge_attr=edge_attr,
                        y=torch.tensor([Tms[i]], dtype=torch.float), pos=pos,
                        len_seq=torch.tensor((seq.shape[0]), dtype=torch.long),
                            torsion_angle=torsion_angle)
            data_list.append(data_3d)
            pbar.update(1)
    return data_list


if __name__ == '__main__':
    load_pyg_data()
