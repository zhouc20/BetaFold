import torch
from typing import  Dict, List

'''
氨基酸序列的编码方法和压缩方法
编码方法得到固定大小的全蛋白质的特征向量，压缩方法提高维度压缩序列长度
氨基酸理化性质以给定，整数0-19编码氨基酸类别，给定默认编码方式，接受其它编码方式，需要用{'class':int}给出
氨基酸种类缩写：
1、Gly缩写：G，甘氨酸 亲水性，分子量：75.07

2、Ala缩写：A，丙氨酸 疏水性，分子量：89.09

3、Val缩写：V，缬氨酸 疏水性，分子量：117.15

4、Leu缩写：L，亮氨酸 疏水性，分子量：131.17

5、Ile缩写：I，异亮氨酸 疏水性，分子量：131.17

6、Phe缩写：F，苯丙氨酸 疏水性，分子量：165.19

7、Trp缩写：W，色氨酸 疏水性，分子量：204.23

8、Tyr缩写：Y，酪氨酸 亲水性，分子量：181.19

9、Asp缩写：D，天冬氨酸 酸性，分子量：33.10

10、Asn缩写：N，天冬酰胺 亲水性，分子量：132.12

11、Glu缩写：E，谷氨酸 酸性，分子量：147.13

12、Lys缩写：K，赖氨酸 碱性，分子量：146.19

13、Gln缩写：Q，谷氨酰胺 亲水性，分子量：146.15

14、Met缩写：M，甲硫氨酸 疏水性，分子量：149.21

15、Ser缩写：S，丝氨酸 亲水性，分子量：105.095.68

16、Thr缩写：T，苏氨酸 亲水性，分子量：119.12

17、Cys缩写：C，半胱氨酸 亲水性，分子量：121.16

18、Pro缩写：P，脯氨酸 疏水性，分子量：115.13

19、His缩写：H，组氨酸 碱性，分子量：155.16

20、Arg缩写：R，精氨酸 碱性，分子量：174.20
'''

init_dict = {'A':0,'G':1,'V':2,'I':3,'L':4,'F':5,'P':6,'Y':7,'M':8,'T':9,'S':10,'H':11,'N':12,'Q':13,'W':14,'R':15,'K':16,'D':17,'E':18,'C':19}


def Amoid_Cluster_decode(Amoid_Dict:Dict):
    dict_base = ['A','G','V','I','L','F','P','Y','M','T','S','H','N','Q','W','R','K','D','E','C']

    Cluster_list = []
    Cluster_list_1 = [Amoid_Dict[dict_base[0]],Amoid_Dict[dict_base[1]],Amoid_Dict[dict_base[2]]]
    Cluster_list_2 =[Amoid_Dict[dict_base[3]], Amoid_Dict[dict_base[4]], Amoid_Dict[dict_base[5]], Amoid_Dict[dict_base[6]]]
    Cluster_list_3 = [Amoid_Dict[dict_base[7]], Amoid_Dict[dict_base[8]], Amoid_Dict[dict_base[9]], Amoid_Dict[dict_base[10]]]
    Cluster_list_4 = [Amoid_Dict[dict_base[11]], Amoid_Dict[dict_base[12]], Amoid_Dict[dict_base[13]], Amoid_Dict[dict_base[14]]]
    Cluster_list_5 = [Amoid_Dict[dict_base[15]], Amoid_Dict[dict_base[16]]]
    Cluster_list_6 = [Amoid_Dict[dict_base[17]], Amoid_Dict[dict_base[18]]]
    Cluster_list_7 =  [Amoid_Dict[dict_base[19]]]

    Cluster_list.append(Cluster_list_1)
    Cluster_list.append(Cluster_list_2)
    Cluster_list.append(Cluster_list_3)
    Cluster_list.append(Cluster_list_4)
    Cluster_list.append(Cluster_list_5)
    Cluster_list.append(Cluster_list_6)
    Cluster_list.append(Cluster_list_7)

    return Cluster_list

init_Cluster_lst = Amoid_Cluster_decode(init_dict)

#AAC方法将氨基酸序列生成整体向量,输入氨基酸序列[N,1],输出[20,1]的特征向量
def AAC_encode_aci_seq(Aci_seq:torch.Tensor,class_num:int=20):
    AAC  = torch.zeros((class_num,1))
    for  i in range(class_num):
        AAC[i] = ((Aci_seq==i).long()).sum()

    AAC = AAC/(AAC.sum())
    return AAC



#CT组合三联体法以一个氨基酸及其左右氨基酸为单位，将20个氨基酸分成7个不同的簇
def CT_encode_aci_seq(Aci_seq:torch.Tensor,Cluster_list:List=init_Cluster_lst):
    Aci_seq_pre = Aci_seq[:-2,:]
    Aci_seq_mid = Aci_seq[1:-1, :]
    Aci_seq_lst = Aci_seq[2:, :]

    Aci_CT = torch.cat((Aci_seq_pre,Aci_seq_mid,Aci_seq_lst),dim=1)
    for i,cluster in enumerate(Cluster_list):
        for j in cluster:
            Aci_CT[Aci_CT == j]=i

    Aci_CT = Aci_CT[:,0]*7*7 + Aci_CT[:,1]*7 + Aci_CT[:,2]

    CT = AAC_encode_aci_seq(Aci_CT,class_num=7*7*7)
    return CT

#输入氨基酸序列[N,1],压缩为[N/3,1]的特征向量
def CT_compress_aci_seq(Aci_seq:torch.Tensor,Cluster_list:List=init_Cluster_lst):
    Aci_seq_pre = Aci_seq[:-2,:]
    Aci_seq_mid = Aci_seq[1:-1, :]
    Aci_seq_lst = Aci_seq[2:, :]

    Aci_CT = torch.cat((Aci_seq_pre,Aci_seq_mid,Aci_seq_lst),dim=1)
    for i,cluster in enumerate(Cluster_list):
        for j in cluster:
            Aci_CT[Aci_CT == j]=i

    Aci_CT = Aci_CT[:,0]*7*7 + Aci_CT[:,1]*7 + Aci_CT[:,2]


#Auto covariance (AC)编码方式，自协方差法主要考虑氨基酸的邻近效应。
# 氨基酸与周围固定数量氨基酸的相互作用表现为疏水性(H1)、亲水性(H2)、净电荷指数(NCI)、极性(P1)、极化率(P2)、侧链。
# 将氨基酸序列替换为6个理化性质的初始值，并归一化为零均值和单位标准差(SD)。

#生成整数与氨基酸序列的对应关系
def Amoid_Dict_decoder(Amoid_Dict:Dict):
    dict_base = ['A', 'G', 'V', 'I', 'L', 'F', 'P', 'Y', 'M', 'T', 'S', 'H', 'N', 'Q', 'W', 'R', 'K', 'D', 'E', 'C']

    Amoid_Int_map = []
    for aci in dict_base:
        Amoid_Int_map.append(Amoid_Dict[aci])
    return Amoid_Int_map

init_map = Amoid_Dict_decoder(init_dict)

def AC_Character_change(Aci_seq:torch.Tensor,Amoid_Int_map:List = init_map):
    Aci_seq = Aci_seq.long()
    H1 = [1.8, -0.4, 4.2, 4.5, 3.8, 2.8, -1.6, -1.3, 1.9, -0.7, -0.8, -3.2, -3.5, -3.5, -0.9, -4.5, -3.9, -3.5, -3.5,
          2.5]
    H1 = torch.Tensor(H1)
    H2 = -H1
    NCI = [6.01,5.97,5.97,6.02,5.98,5.48,6.48,5.66,5.74,5.87,5.68,7.59,5.41,5.65,5.89,10.76,9.74,2.77,3.22,5.07]
    NCI = torch.Tensor(NCI)
    P1 = [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    P1 = torch.Tensor(P1)
    P2 = [7.8, 7.2, 6.6, 5.3, 9.1, 3.9, 5.2, 3.2, 2.3, 5.9, 6.8, 2.3, 4.3, 4.2, 1.4, 5.1, 5.9, 5.3, 6.3, 1.9]
    P2 = torch.Tensor(P2)

    Amoid_Int_map = torch.tensor(Amoid_Int_map)

    H1 = H1[Amoid_Int_map]
    H2 = H2[Amoid_Int_map]
    NCI = NCI[Amoid_Int_map]
    P1 = P1[Amoid_Int_map]
    P2 = P2[Amoid_Int_map]

    Aci_H1 = H1[Aci_seq]
    Aci_H2 = H2[Aci_seq]
    Aci_NCI = NCI[Aci_seq]
    Aci_P1 = P1[Aci_seq]
    Aci_P2 = P2[Aci_seq]

    AC = torch.cat((Aci_H1,Aci_H2,Aci_NCI,Aci_P1,Aci_P2,Aci_seq)).float()

    return AC

#输入氨基酸序列[N,1],输出[lag,6]的特征向量
def AC_encode(Aci_seq:torch.Tensor,Amoid_Int_map:List = init_map,lag:int = 30):
    AC = AC_Character_change(Aci_seq,Amoid_Int_map)
    Aci_len = AC.shape[0]

    AC_mean = AC.mean(dim = 0)
    AC_std = AC.std(dim = 0)

    AC = (AC-AC_mean)/AC_std
    AC_ = torch.zeros(lag,6)
    for l in range(1,lag):
        for j in range(6):
            AC_pre = AC[:-l,:]
            AC_lst = AC[l:,:]
            AC_[l][j] = ((AC_pre-AC.mean(dim = 0))*(AC_lst-AC.mean(dim = 0))).sum()/(Aci_len-l)

    return AC_


#输入氨基酸序列[N,1],压缩为[N-2*lag,6]的特征向量
def AC_compress(Aci_seq: torch.Tensor, Amoid_Int_map: List = init_map, lag:int = 30):
    AC = AC_Character_change(Aci_seq, Amoid_Int_map)
    Aci_len = AC.shape[0]

    AC_mean = AC.mean(dim=0)
    AC_std = AC.std(dim=0)

    AC = (AC - AC_mean) / AC_std
    AC_ = torch.zeros(Aci_len - 2*lag, 6)
    base = AC[lag:-lag,:]

    for l in range(2*lag):
        AC_ += base*AC[l:-2*lag+l,:]
    AC_ = AC_/(2*lag)

    return AC_