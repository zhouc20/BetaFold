from Bio.PDB import *
import numpy as np



def get_PDB_data(data_path):
    C_a_list = np.ones((0,3))
    C_list = np.ones((0,3))
    N_list = np.ones((0,3))
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    data_id = data_path.split('/')[-1].replace(".pdb", "")
    try:
        data = parser.get_structure(data_id, data_path)
    except:
        return C_a_list, C_list, N_list
    model = data.get_models()
    models = list(model)
    for md in models:
        chain = list(md.get_chains())
        for ch in chain:
            redisure = list(ch.get_residues())
            for rs in redisure:
                atom = list(rs.get_atoms())
                if len(atom)<7:
                    continue
                N = atom[0]
                C_a = atom[1]
                C = atom[2]
                C_a_list = np.concatenate((C_a_list,C_a.get_coord().reshape((1,-1))),axis=0)
                N_list = np.concatenate((N_list, N.get_coord().reshape((1,-1))), axis=0)
                C_list = np.concatenate((C_list, C.get_coord().reshape((1,-1))), axis=0)

    return C_a_list,C_list,N_list










