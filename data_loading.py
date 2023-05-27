import numpy as np
import glob
import csv


def read_csvs():
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
    files = glob.glob('./ProTstab2EachSpeciesDatasets/*.csv')
    data = []
    for file in files:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                data.append(row)
    data_array = np.array(data)
    return data_array


if __name__ == '__main__':
    read_csvs()
