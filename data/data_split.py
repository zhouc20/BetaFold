import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_data():
    # 数据读取
    for filepath, dirnames, filenames in os.walk(r'ProTstab2EachSpeciesDatasets'):
        for filename in filenames:
            data = pd.read_csv(os.path.join(filepath, filename), sep=',', header=None)
            nums = data.iloc[:, 0]
            nums = np.array(nums)
            train, test = train_test_split(nums, test_size=0.2, train_size=0.8)
            test, valid = train_test_split(test, test_size=0.5, train_size=0.5)
            print(filename, len(train), len(valid), len(test))

            path_name = os.path.join(filepath, filename.split('.')[0]+"_train.txt")
            np.savetxt(path_name, train)
            path_name = os.path.join(filepath, filename.split('.')[0] + "_test.txt")
            np.savetxt(path_name, test)
            path_name = os.path.join(filepath, filename.split('.')[0] + "_valid.txt")
            np.savetxt(path_name, valid)
            # print(pd.read_csv(path_name))


if __name__ == "__main__":
    split_data()


