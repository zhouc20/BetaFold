import csv
import glob
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


"""
蛋白质结构数据集中所有可能的atom_name, 共36种
C, CA, CB, CD, CD1, CD2, CE, CE1, CE2, CE3, CG, CG1, CG2, CH2, CZ, CZ2, CZ3, 
N, ND1, ND2, NE, NE1, NE2, NH1, NH2, NZ, 
O, OD1, OD2, OE1, OE2, OG, OG1, OH, 
SD, SG

数据集蛋白质平均长度约为250, 也即每条蛋白质数据约250个CA

数据集中共20种氨基酸, 分别为
ALA, ARG, ASN, ASP, CYS, 
GLN, GLU, GLY, HIS, ILE, 
LEU, LYS, MET, PHE, PRO, 
SER, THR, TRP, TYR, VAL
"""

RESIDUE_TO_NUM = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

ATOM_TO_NUM = {
    'C': 0, 'CA': 1, 'CB': 2, 'CD': 3, 'CD1': 4, 'CD2': 5, 
    'CE': 6, 'CE1': 7, 'CE2': 8, 'CE3': 9, 
    'CG': 10, 'CG1': 11, 'CG2': 12, 'CH2': 13, 
    'CZ': 14, 'CZ2': 15, 'CZ3': 16, 
    'N': 17, 'ND1': 18, 'ND2': 19, 
    'NE': 20, 'NE1': 21, 'NE2': 22, 'NH1': 23, 'NH2': 24, 'NZ': 25, 
    'O': 26, 'OD1': 27, 'OD2': 28, 
    'OE1': 29, 'OE2': 30, 'OG': 31, 'OG1': 32, 'OH': 33, 
    'SD': 34, 'SG': 35
}


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def read_csvs(data_filter=lambda _: True):
    """
    FASTA格式与氨基酸的对应关系:
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
    files = glob.glob('./ProTstab2_dataset_new/*.csv')
    csvs_data = {}
    for file in files:
        csv_data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                if data_filter(row):
                    csv_data.append(row)
        csvs_data[os.path.splitext(os.path.split(file)[-1])[0]] = np.array(csv_data)
    return csvs_data


class StructDataset(object):
    def __init__(self, path, batch_size=4, random=True, node_filter='CA'):
        super().__init__()
        print('loading...')
        self.data = pickle_load(path)
        print('loading finish')
        # self.numerical_features()
        self.Tms = [float(v['Tm']) for v in self.data.values()]
        self.structures = [v['structure'] for v in self.data.values()]
        del self.data # free memory
        self.atom_names =[v['atom_name'] for v in self.structures]
        self.xs = [v['x'] for v in self.structures]
        self.ys = [v['y'] for v in self.structures]
        self.zs = [v['z'] for v in self.structures]
        self.residue_names = [v['residue_name'] for v in self.structures]
        
        self.random = random
        self.batch_size = batch_size
        self.num_getitem = 0  # 计数调用__getitem__的次数
        self.order = np.random.permutation(len(self.structures))
        
        self.node_filter = node_filter  # 'CA' | 'all'
        
        
    def numerical_features(self):
        dist = []
        for v in self.data.values():
            vv = v['structure']
            indices = [i for i, x in enumerate(vv['atom_name']) if x == 'CA']
            x, y, z = np.array(vv['x']), np.array(vv['y']), np.array(vv['z'])
            x, y, z = x[indices], y[indices], z[indices]
            d = ((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2 + (z[:-1] - z[1:]) ** 2) ** 0.5
            d = np.mean(d)
            dist.append(d)
        print(sum(dist) / len(dist))

    def __len__(self):
        return len(self.structures) // self.batch_size

    def __getitem__(self, item):
        pos = []
        batch = []
        node_atom = []
        
        self.num_getitem += self.batch_size
        if self.num_getitem > len(self):
            self.order = np.array(random.sample(range(len(self.structures)), len(self.structures)))
            self.num_getitem -= len(self)
        if self.random:
            idxes = self.order[range(item * self.batch_size, (item + 1) * self.batch_size)]
        else:
            idxes = np.array(range(item * self.batch_size, (item + 1) * self.batch_size))
        
        if self.node_filter == 'CA':
            for b, i in enumerate(idxes):
                js = [k for k, name in enumerate(self.atom_names[i]) if name == 'CA']
                x = torch.tensor([self.xs[i][j] for j in js], dtype=torch.float32)
                y = torch.tensor([self.ys[i][j] for j in js], dtype=torch.float32)
                z = torch.tensor([self.zs[i][j] for j in js], dtype=torch.float32)
                residue_name = [self.residue_names[i][j] for j in js]
                residue_name = list(map(lambda inputs: RESIDUE_TO_NUM[inputs], residue_name))
                residue_name = torch.tensor(residue_name, dtype=torch.int64)
                pos.append(torch.stack([x, y, z], dim=-1))
                batch.append(torch.tensor([b] * x.shape[0], dtype=torch.int64))
                node_atom.append(residue_name)
        elif self.node_filter == 'all':
            for b, i in enumerate(idxes):
                x = torch.tensor(self.xs[i], dtype=torch.float32)
                y = torch.tensor(self.ys[i], dtype=torch.float32)
                z = torch.tensor(self.zs[i], dtype=torch.float32)
                atom_name = list(map(lambda inputs: ATOM_TO_NUM[inputs], self.atom_names[i]))
                atom_name = torch.tensor(atom_name, dtype=torch.int64)
                pos.append(torch.stack([x, y, z], dim=-1))
                batch.append(torch.tensor([b] * x.shape[0], dtype=torch.int64))
                node_atom.append(atom_name)
                
        return {'pos': torch.cat(pos, dim=0).clone(),
                'batch': torch.cat(batch, dim=0).clone(),
                'node_atom': torch.cat(node_atom, dim=0).clone(),
                'Tm': torch.tensor([self.Tms[i] for i in idxes], dtype=torch.float32).clone().unsqueeze(-1)}


def main():
    # all_data = read_csvs()
    # print(all_data.keys())
    d = StructDataset(path='./BetaFold/StructuredDatasets/train_dataset.pkl')
    for i in range(len(d)):
        data = d[i]
        print(data['pos'].shape)
        print(data['batch'].shape)
        print(data['node_atom'].shape)
        print(data['Tm'].shape)


if __name__ == '__main__':
    main()
