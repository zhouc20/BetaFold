import numpy as np
import torch

def angle_encode(angle):
    angle = np.array(angle)
    angle1 = np.sin(angle)
    angle2 = np.sin(2*angle)
    angle3 = np.sin(4*angle)
    angle4 = np.cos(angle)
    angle5 = np.cos(2*angle)
    angle6 = np.cos(4*angle)
    encode_angle = np.concatenate((angle1,angle2,angle3,angle4,angle5,angle6),axis=1)
    encode_angle = torch.from_numpy(encode_angle)
    return encode_angle


