
import numpy as np
from data.geometry import *
from data.PDB_analyze import get_PDB_data

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

def Turn_site_found(C_a,max_dis):
    site_pair_list = np.ones((0, 2))
    len = C_a.shape[0]
    for offset in range(2,5):
        site_index = np.arange(0, len - offset)

        seq1 = C_a[:len - offset, :]
        seq2 = C_a[offset:, :]
        dis = np.sqrt(((seq1 - seq2) ** 2).sum(1))

        index1 = site_index[dis < max_dis]

        index2 = index1 + offset

        pair_site = np.concatenate((index1.reshape((-1, 1)), index2.reshape((-1, 1))), axis=1)
        site_pair_list = np.concatenate((site_pair_list, pair_site), axis=0)

    site_pair_list = site_pair_list.astype(int)



    return site_pair_list

def Loop_site_found(C_a,max_dis = 9.5):
    site_pair_list = np.ones((0, 2))
    len = C_a.shape[0]
    offset = 7
    site_index = np.arange(0, len - offset)
    seq1 = C_a[:len - offset]
    seq2 = C_a[offset:]
    dis = np.sqrt(((seq1 - seq2) ** 2).sum(1))
    pair = dis < max_dis
    index1 = site_index[pair]
    index2 = index1 + offset
    pair_site = np.cat((index1, index2), axis=1)
    site_pair_list = np.cat((site_pair_list, pair_site), axis=0)



    return site_pair_list

def Beta_turn(C_a,N,C,length,max_dis = 9.5):
    phi, puci = portein_Dihedral_angle(C_a, N, C, length)
    site_list = Turn_site_found(C_a,max_dis)

    site1 = site_list[:,0]
    site2 = site_list[:,1]

    phi1 = phi[site1].reshape(1,-1)
    phi2 = phi[site2].reshape(1,-1)
    phi3 = phi[site1+1].reshape(1,-1)
    phi4 = phi[site2+1].reshape(1,-1)
    puci1 =  puci[site1].reshape(1,-1)
    puci2 = puci[site2].reshape(1,-1)
    puci3 = puci[(site1-1)].reshape(1,-1)
    puci4 = puci[(site2-1)].reshape(1,-1)

    angle_sets = np.concatenate((puci3,phi1,puci1,phi3,puci4,phi2,puci2,phi4),axis=0)
    angle_sets = angle_sets.T


    cor1 = C_a[site1]
    cor2 = C_a[site2]

    tran = translate(cor1,cor2)
    rot = rotation(cor1,cor2)
    obs_per = np.concatenate((tran, rot, angle_sets), axis=1)

    obs_ref_per = np.concatenate((puci[:-1].reshape(-1,1), phi[:-1].reshape(-1,1), puci[1:].reshape(-1,1), phi[1:].reshape(-1,1),), axis=1)

    return obs_per,obs_ref_per,site_list

def Beta_turn_(C_a,N,C,site_list,length):
    phi, puci = portein_Dihedral_angle(C_a, N, C, length)

    site1 = site_list[:,0]
    site2 = site_list[:,1]

    phi1 = phi[site1].reshape(1,-1)
    phi2 = phi[site2].reshape(1,-1)
    phi3 = phi[site1+1].reshape(1,-1)
    phi4 = phi[site2+1].reshape(1,-1)
    puci1 =  puci[site1].reshape(1,-1)
    puci2 = puci[site2].reshape(1,-1)
    puci3 = puci[(site1-1)].reshape(1,-1)
    puci4 = puci[(site2-1)].reshape(1,-1)

    angle_sets = np.concatenate((puci3,phi1,puci1,phi3,puci4,phi2,puci2,phi4),axis=0)
    angle_sets = angle_sets.T


    cor1 = C_a[site1]
    cor2 = C_a[site2]

    tran = translate(cor1,cor2)
    rot = rotation(cor1,cor2)
    obs_per = np.concatenate((tran, rot, angle_sets), axis=1)

    obs_ref_per = np.concatenate((puci[:-1].reshape(-1,1), phi[:-1].reshape(-1,1), puci[1:].reshape(-1,1), phi[1:].reshape(-1,1),), axis=1)

    return obs_per,obs_ref_per

def Omega_Loop(C_a,N,C,length,max_dis = 9.5):
    phi, puci = portein_Dihedral_angle(C_a, N, C, length)
    site_list = Loop_site_found(C_a,max_dis)
    site1 = site_list[:,0].reshape(1,-1)
    site2 = site_list[:, 1].reshape(1, -1)
    phi1 = phi[site1].reshape(1,-1)
    phi2 = phi[site1+1].reshape(1,-1)
    phi3 = phi[site1+2].reshape(1,-1)
    phi4 = phi[site1+3].reshape(1,-1)
    phi5 = phi[site1 + 4].reshape(1, -1)
    phi6 = phi[site1 + 5].reshape(1, -1)
    phi7 = phi[site1 + 6].reshape(1, -1)
    phi8 = phi[site1 + 7].reshape(1, -1)
    puci1 =  puci[site1].reshape(1,-1)
    puci2 = puci[site1+1].reshape(1,-1)
    puci3 = puci[(site1+2)].reshape(1,-1)
    puci4 = puci[(site1+3)].reshape(1,-1)
    puci5 = puci[site1 + 4].reshape(1, -1)
    puci6 = puci[site1 + 5].reshape(1, -1)
    puci7 = puci[(site1 + 6)].reshape(1, -1)
    puci8 = puci[(site1 + 7)].reshape(1, -1)


    angle_sets = np.concatenate((puci1,phi1,puci2,phi2,puci3,phi3,puci4,phi4,puci5,phi5,puci6,phi6,puci7,phi7,puci8,phi8),axis=0)

    cor1 = C_a[site1]
    cor2 = C_a[site2]

    tran = translate(cor1,cor2)
    rot = rotation(cor1,cor2)
    obs_per = np.concatenate((tran,rot,angle_sets), axis=1)
    obs_ref_per = np.concatenate((puci, phi), axis=1)


    return obs_per,obs_ref_per

def Dihedral_angle_5(C_a,N,C,length):
    phi, puci = portein_Dihedral_angle(C_a,N,C,length)
    obs_per = np.ones((length-5,0))


    for i in range(5):
        obs_per = np.concatenate((obs_per, puci[i:length - 5 + i].reshape(-1,1)),axis=1)
        obs_per = np.concatenate((obs_per,phi[i:length-5+i].reshape(-1,1)),axis=1)

    obs_ref_per =  np.concatenate((puci.reshape(-1,1),phi.reshape(-1,1)),axis=1)

    return obs_per,obs_ref_per

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
    obs_ref_per = np.concatenate((puci[:-1].reshape(-1,1), phi[:-1].reshape(-1,1), puci[1:].reshape(-1,1), phi[1:].reshape(-1,1)), axis=1)

    return obs_per,obs_ref_per


def feature_deal(data_path,cls,max_dis = 9.5):
    C_a_list, C_list, N_list = get_PDB_data(data_path)


    length = C_a_list.shape[0]
    try:
        if cls == 'phi_puci_5':
            if length < 6:
                return np.ones((0, 1)), np.ones((0, 1))
            return Dihedral_angle_5(C_a_list, N_list, C_list, length)

        elif cls == 'site_pair':
            return Site_pair(C_a_list, N_list, C_list, length, max_dis)

        elif cls == 'beta_turn':
            return Beta_turn(C_a_list, N_list, C_list, length, max_dis)

        elif cls == 'omega_loop':
            return Omega_Loop(C_a_list, N_list, C_list, length, max_dis)
    except:
        return np.ones((0, 1)), np.ones((0, 1))


    return

def feature_deal_(C_a_list, C_list, N_list ,cls,max_dis = 9.5):



    length = C_a_list.shape[0]
    try:
        if cls == 'phi_puci_5':
            if length < 6:
                return np.ones((0, 1)), np.ones((0, 1))
            return Dihedral_angle_5(C_a_list, N_list, C_list, length)

        elif cls == 'site_pair':
            return Site_pair(C_a_list, N_list, C_list, length, max_dis)

        elif cls == 'beta_turn':
            return Beta_turn(C_a_list, N_list, C_list, length, max_dis)

        elif cls == 'omega_loop':
            return Omega_Loop(C_a_list, N_list, C_list, length, max_dis)
    except:
        return np.ones((0, 1)), np.ones((0, 1))


    return
