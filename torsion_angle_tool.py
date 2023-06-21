import torch
from data_loading import pickle_load


FASTA_LOOKUP = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL'
}

LOOKUP = {
        'ALA': [],
        'ARG': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD'],
                ['CB', 'CG', 'CD', 'NE'],
                ['CG', 'CD', 'NE', 'CZ']],
        'ASN': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'OD1']],
        'ASP': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'OD1']],
        'CYS': [['N', 'CA', 'CB', 'SG']],
        'GLN': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD'],
                ['CB', 'CG', 'CD', 'OE1']],
        'GLU': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD'],
                ['CB', 'CG', 'CD', 'OE1']],
        'GLY': [],
        'HIS': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'ND1']],
        'ILE': [['N', 'CA', 'CB', 'CG1'],
                ['CA', 'CB', 'CG1', 'CD1']],
        'LEU': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD1']],
        'LYS': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD'],
                ['CB', 'CG', 'CD', 'CE'],
                ['CG', 'CD', 'CE', 'NZ']],
        'MET': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'SD'],
                ['CB', 'CG', 'SD', 'CE']],
        'PHE': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD1']],
        'PRO': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD']],
        'SER': [['N', 'CA', 'CB', 'OG']],
        'THR': [['N', 'CA', 'CB', 'OG1']],
        'TRP': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD1']],
        'TYR': [['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD1']],
        'VAL': [['N', 'CA', 'CB', 'CG1']],
    }


def batch_dot(v1, v2):
    """批量化点积

    Args:
        v1 (FloatTensor): vector1, shape of (B, d)
        v2 (FloatTensor): vector2, shape of (B, d)
    Returns:
        FloatTensor: result, shape of (B, d)
    """
    dot_product = torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).squeeze()
    return dot_product


def points2plane(p1, p2, p3):
    """从三个点的坐标计算平面方程

    Args:
        p1 (FloatTensor): point1 coordinates, shape of (3,)
        p2 (FloatTensor): point2 coordinates, shape of (3,)
        p3 (FloatTensor): point3 coordinates, shape of (3,)
        
    Returns:
        FloatTensor: plane function, shape of (4,), ax+by+cz+d=0, (a, b, c, d)
    """
    vector1 = p2 - p1
    vector2 = p3 - p1
    # 计算法向量（即两个向量的叉积）
    normal_vector = torch.cross(vector1, vector2)
    # 提取法向量的分量
    a, b, c = normal_vector
    # 提取参考点
    x0, y0, z0 = p1
    # 构建平面方程字符串
    # plane_equation = f"{a} * (x - {x0}) + {b} * (y - {y0}) + {c} * (z - {z0}) = 0"
    # print("平面方程：", plane_equation)
    # 整理为ax+by+cz+d=0的形式
    s = torch.tensor([a, b, c, - a * x0 - b * y0 - c * z0], dtype=torch.float32, device=p1.device)
    return s


def planes2angle(s1, s2):
    """从两个平面的方程计算二面角

    Args:
        s1 (FloatTensor): plane1 function, shape of (4,), ax+by+cz+d=0, (a, b, c, d)
        s2 (FloatTensor): plane1 function, shape of (4,), ax+by+cz+d=0, (a, b, c, d)
        
    Returns:
        FloatTensor: angle, shape of (1,), [0, pi]
    """
    a1, b1, c1, d1 = s1
    a2, b2, c2, d2 = s2
    # 计算两个平面的法向量
    normal_vector1 = s1[:3]
    normal_vector2 = s2[:3]
    # 计算二面角
    cos_angle = torch.dot(normal_vector1, normal_vector2) / (torch.norm(normal_vector1) * torch.norm(normal_vector2))
    angle = torch.acos(cos_angle)
    # angle为弧度, [0, pi]
    return angle

def points2angle(p1, p2, p3, p4):
    """从四个点的坐标计算二面角, 也即前三个点的平面和后三个点的平面构成的夹角

    Args:
        p1 (FloatTensor): point1 coordinates, shape of (3,)
        p2 (FloatTensor): point2 coordinates, shape of (3,)
        p3 (FloatTensor): point3 coordinates, shape of (3,)
        p4 (FloatTensor): point4 coordinates, shape of (3,)

    Returns:
        FloatTensor: shape of (1,)
    """
    s1 = points2plane(p1, p2, p3)
    s2 = points2plane(p2, p3, p4)
    angle = planes2angle(s1, s2)
    return angle.unsqueeze(0)


def side_chain_torsion_angles(fasta, structure, pad=-1.0):
    """计算侧链扭转角

    Args:
        fasta (str): FASTA序列
        structure (dict): 蛋白质结构数据
        pad (float): 填充值

    Returns:
        FloatTensor: shape of (N, 4)
    """
    result = []
    for i, residue_type in enumerate(fasta):
        query = LOOKUP[FASTA_LOOKUP[residue_type]]
        tmp = [torch.tensor([pad]), torch.tensor([pad]), torch.tensor([pad]), torch.tensor([pad])]
        slice = [k for k in range(len(structure['residue_idx'])) if structure['residue_idx'][k] == i + 1]
        for j, q in enumerate(query):
            idxes = [structure['atom_name'][slice[0]:slice[-1] + 1].index(k) for k in q]
            x = [structure['x'][slice[0]:slice[-1] + 1][k] for k in idxes]
            y = [structure['y'][slice[0]:slice[-1] + 1][k] for k in idxes]
            z = [structure['z'][slice[0]:slice[-1] + 1][k] for k in idxes]
            p1, p2, p3, p4 = torch.tensor([x, y, z], dtype=torch.float32).permute(1, 0)
            angle = points2angle(p1, p2, p3, p4)
            tmp[j] = angle
        result.append(torch.cat(tmp, dim=0))
    return torch.stack(result, dim=0)


def main():
    d = pickle_load('./StructuredDatasets/test2_dataset.pkl')
    for v in d.values():
        angle = side_chain_torsion_angles(v['simple_fasta'], v['structure'])
        print(angle)


if __name__ == '__main__':
    main()
