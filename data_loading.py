import csv
import glob
import os

import numpy as np


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


def main():
    all_data = read_csvs()
    print(all_data.keys())


if __name__ == '__main__':
    main()
