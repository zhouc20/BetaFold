'''
Author: lin-yh20@mails.tsinghua.edu.cn
Date: 2023-04-24
LastEditTime: 2023-04-24
LastEditors: Yuhuan Lin
Description: Get Geometry Feature from Protein Ca,N,C
FilePath: /model/Attention_Module.py
'''



from geometry import *

def portein_Dihedral_angle(C_a,N,C,length):
    aci_len = length
    phi = np.ones(0)
    puci =  np.ones(0)
    for i in range(aci_len-1):
        N1 = N[i]
        Ca = C_a[i]
        C1 = C[i]
        N2 = N[i+1]
        C2 = C[i+1]
        puci_tmp =np.array(Dihedral_angle(N2,Ca,C1,N1)).reshape((1))
        phi_tmp = np.array(Dihedral_angle(C2,N2,Ca,C1)).reshape((1))
        phi = np.concatenate((phi,phi_tmp))
        puci = np.concatenate((puci, puci_tmp))

    return phi,puci

def site_pair_found(C_a,max_dis):
    site_pair_list = np.ones((0,2))
    C_a = C_a[2:-2]
    len = C_a.shape[0]


    for offset in range(len-1):
        if offset<5 :
            continue
        site_index = np.arange(0,len-offset)
        seq1 = C_a[:len-offset]
        seq2 = C_a[offset:]

        dis = np.sqrt(((seq1-seq2)**2).sum(1))

        pair = dis<max_dis
        index1 = (site_index[pair]).reshape(-1,1)
        index2 = (index1 + offset).reshape(-1,1)
        pair_site = np.concatenate((index1,index2),axis = 1)
        site_pair_list = np.concatenate((site_pair_list,pair_site),axis = 0)
    site_pair_list += 2
    return site_pair_list

def Dihedral_angle_5(C_a,N,C,length):
    phi, puci = portein_Dihedral_angle(C_a,N,C,length)
    obs_per = np.ones((length-5,0))


    for i in range(5):
        obs_per = np.concatenate((obs_per, puci[i:length - 5 + i].reshape(-1,1)),axis=1)
        obs_per = np.concatenate((obs_per,phi[i:length-5+i].reshape(-1,1)),axis=1)


    return obs_per

def Site_pair(C_a,N,C,length,max_dis = 9.5):
    phi, puci = portein_Dihedral_angle(C_a, N, C, length)
    site_list = site_pair_found(C_a,max_dis)
    site1 = site_list[:,0].astype(int)
    site2 = site_list[:,1].astype(int)
    print(site1)
    print(site2)
    phi1 = phi[site1].reshape(-1,1)
    phi2 = phi[site2].reshape(-1,1)
    phi3 = phi[site1+1].reshape(-1,1)
    phi4 = phi[site2+1].reshape(-1,1)
    puci1 =  puci[site1].reshape(-1,1)
    puci2 = puci[site2].reshape(-1,1)
    puci3 = puci[(site1-1)].reshape(-1,1)
    puci4 = puci[(site2-1)].reshape(-1,1)

    angle_sets = np.concatenate((puci3,phi1,puci1,phi3,puci4,phi2,puci2,phi4),axis=1)

    cor1 = C_a[site1]
    cor2 = C_a[site2]

    tran = translate(cor1,cor2)
    print(tran.shape)
    rot = rotation(cor1,cor2)
    print(rot.shape)
    obs_per = np.concatenate((tran,rot,angle_sets),axis=1)

    return obs_per
