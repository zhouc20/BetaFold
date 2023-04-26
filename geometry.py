'''
Author: lin-yh20@mails.tsinghua.edu.cn
Date: 2023-04-24
LastEditTime: 2023-04-24
LastEditors: Yuhuan Lin
Description: Geometry Function for Protein
FilePath: /model/Attention_Module.py
'''


import numpy as np
import math


def Dihedral_angle(A, B, C, D):
    a, b, c = B - A, C - B, D - C
    n1, n2 = np.cross(a, b), np.cross(b, c)
    return math.acos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))) * 180 / math.pi

def distance(A,B):
    dis = ((A-B)**2).sum()
    return dis

def translate(A,B):
    return (A-B)

def rotation(A,B):

    x = (A-B)[:,0:1]
    y = (A-B)[:,1:2]
    z = (A-B)[:,2:3]

    alpha = np.arctan(y/x)
    beta = np.arctan(z/x)
    theta = np.arctan(z/y)
    ori = np.concatenate((alpha,beta,theta),axis = 1)
    return ori




