import csv
import glob
import os
import pickle

import numpy as np
from torch.utils.data import Dataset, DataLoader


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


class StructDataset(Dataset):
    def __init__(self):
        super().__init__()
        print('loading...')
        self.data = pickle_load('./BetaFold/StructuredDatasets/train_dataset.pkl')
        print('loading finish')
        self.numerical_features()
        
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
        return 0

    def __getitem__(self, item):
        return 0


def main():
    # all_data = read_csvs()
    # print(all_data.keys())
    d = StructDataset()


if __name__ == '__main__':
    main()
