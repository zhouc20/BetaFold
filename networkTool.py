'''
Author: lin-yh20@mails.tsinghua.edu.cn
Date: 2023-04-24
LastEditTime: 2023-04-24
LastEditors: Yuhuan Lin
Description: Network parameters
FilePath: /model/networkTool.py
'''

import torch
import os, random
import numpy as np

# torch.set_default_tensor_type(torch.DoubleTensor)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network parameters
embed1 = 128
embed2 = 128
embed3 = 128
levelNumK = 4


expName = './Exp/Obj'
DataRoot = './Data/Obj'
checkpointPath = expName + '/checkpoint'

trainDataRoot = DataRoot + "/train/*.mat"  # DON'T FORGET RUN ImageFolder.calcdataLenPerFile() FIRST

# Random seed
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["H5PY_DEFAULT_READONLY"] = "1"
