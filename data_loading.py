import numpy as np
import glob
import csv


def read_csvs():
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
