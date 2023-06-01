import json
import os.path
import time

import requests
from tqdm import tqdm
import numpy as np

from data_loading import read_csvs

url = 'https://api.esmatlas.com/foldSequence/v1/pdb'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.41',
    'Accept': 'application/json, text/plain, */*',
    'origin': 'https://esmatlas.com',
    'content-type': 'text/plain'}

csvs_data = read_csvs(data_filter=lambda x: len(x[2]) <= 400)
csv_name, extra_data = None, None
try:
    for csv_name, csv_data in csvs_data.items():
        # 筛选物种
        all_species = ('A.thaliana', 'B.subtilis', 'C.elegans', 'D.melanogaster',
                       'D.rerio', 'E.coli', 'G.stearothermophilus', 'H.sapiens',
                       'M.musculus', 'O.antarctica', 'P.torridus', 'S.cerevisiae',
                       'T.thermophilus', 'thermophilus')
        species_filter = ('A.thaliana', 'B.subtilis', 'C.elegans', 'D.melanogaster')
        csv_data = csv_data[np.isin(csv_data[:, 3], species_filter)]
        # 加载上次的断点
        if os.path.exists(f'./StructuredDatasets/{csv_name}.json'):
            with open(f'./StructuredDatasets/{csv_name}.json') as json_file:
                extra_data = json.load(json_file)
        else:
            extra_data = {}
        # 取出剩余条目
        num_rest = len(set(csv_data[:, 1]) - set(extra_data.keys()))
        if num_rest == 0:
            continue
        # 爬数据
        with tqdm(desc=f'downloading {csv_name}', total=num_rest, unit='proteins') as pbar:
            for i in range(len(csv_data)):
                # 原始数据
                protein_id = csv_data[i, 1]
                simple_fasta = csv_data[i, 2]
                species = csv_data[i, 3]
                Tm = csv_data[i, 4]
                # 过长序列不行
                if len(simple_fasta) > 400:
                    print('sequence too long')
                    continue
                # 跳过已爬过的数据
                if protein_id in extra_data.keys():
                    continue
                # 不断尝试爬取直至成功
                wait = 0.5
                while True:
                    result = requests.post(url, data=simple_fasta, headers=headers)
                    result = result.text
                    # 成功爬取的报文以'HEADER'开头
                    if result[:6] == 'HEADER':
                        pbar.set_postfix({'protein_id': protein_id, 'state': 'success'})
                        time.sleep(0.1)
                        break
                    # 失败报文可能包含'INTERNAL SERVER ERROR', 'forbidden'
                    else:
                        wait = wait * 1.2 if wait * 1.2 <= 10 else wait
                        time.sleep(wait)
                        pbar.set_postfix({'protein_id': protein_id, 'state': result})
                # 定位到有效数据区
                cut_idx = result.find('ATOM      1')
                result = result[cut_idx:]
                structure = {'atom_name': [],
                             'residue_name': [],
                             'chain_id': [],
                             'residue_idx': [],
                             'x': [],
                             'y': [],
                             'z': [],
                             'occupancy': [],
                             'temperature_factor': [],
                             'chemical_symbol': []}
                # 保存结构信息
                for j, line in enumerate(result.split('\n')):
                    line = line.replace('-', ' -')
                    parts = line.split(' ')
                    parts = list(filter(lambda x: x != '', parts))
                    assert parts[0] == 'ATOM'
                    assert int(parts[1]) == j + 1
                    structure['atom_name'].append(parts[2])
                    structure['residue_name'].append(parts[3])
                    structure['chain_id'].append(parts[4])
                    structure['residue_idx'].append(int(parts[5]))
                    structure['x'].append(float(parts[6]))
                    structure['y'].append(float(parts[7]))
                    structure['z'].append(float(parts[8]))
                    structure['occupancy'].append(float(parts[9]))
                    structure['temperature_factor'].append(float(parts[10]))
                    structure['chemical_symbol'].append(parts[11])
                # 添加数据
                extra_data[protein_id] = {'simple_fasta': simple_fasta,
                                          'species': species,
                                          'Tm': Tm,
                                          'structure': structure}
                pbar.update(1)
        # 保存为json
        with open(f'./StructuredDatasets/{csv_name}.json', 'w') as json_file:
            json.dump(extra_data, json_file)
finally:
    # 程序意外终止时, 如KeyBoardInterrupt, 则保存已爬取的数据
    # 由于数据量大, json.dump耗时较久, 约十几秒, 切勿强行结束程序, 导致保存中断, 数据丢失
    if csv_name is not None:
        print('saving json file, please do not kill process')
        with open(f'./StructuredDatasets/{csv_name}.json', 'w') as json_file:
            json.dump(extra_data, json_file)
