import time
import numpy as np
import torch.nn as nn
import torch
from data.datalist import DataListSet
from data.pyg_data_loader import load_pyg_data, load_pygdata_3d
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader as PygDataloader
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
import argparse
import random
import os, sys
import math
import errno
import matplotlib.pyplot as plt
from layer.GPS_model import GPSModel
from layer.pronet.pronet import ProNet


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="thermal_stability")
parser.add_argument("--dataset", type=str, default="ProTstab2_3d_full")
parser.add_argument('--model', type=str, default='GPS')
parser.add_argument('--level', type=str, default='aminoacid')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cos_lr', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--loss', type=str, default='L1Loss')
parser.add_argument('--in_dim', type=int, default=1)
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--out_dim', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_layers_MPNN', type=int, default=1)
parser.add_argument('--num_layers_Transformer', type=int, default=0)
parser.add_argument('--num_layers_regression', type=int, default=2)
parser.add_argument('--num_layers_cls', type=int, default=0)
parser.add_argument('--rw_steps', type=int, default=20)
parser.add_argument('--pe_dim', type=int, default=16)
parser.add_argument('--se_type', type=str, default='linear')
parser.add_argument('--se_norm', type=str, default='none')
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--global_pool', type=str, default='add')
parser.add_argument('--local_gnn_type', type=str, default='CustomGatedGCN')
parser.add_argument('--global_model_type', type=str, default='Transformer')
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--pna_degrees', type=int, default=None)
parser.add_argument('--equivstable_pe', action='store_true', default=False)
parser.add_argument('--norm_type', type=str, default='layer')
parser.add_argument('--JK', type=str, default='none')
parser.add_argument('--equiformer', action='store_true', default=False)
parser.add_argument("--radius", type=float, default=5.0)
parser.add_argument("--max_num_neighbors", type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.0, help='drop out rate')
parser.add_argument('--attn_dropout', type=float, default=0.0)
parser.add_argument('--batch_norm', action='store_true', default=False)
parser.add_argument('--layer_norm', action='store_true', default=False)
parser.add_argument('--bigbird_cfg', action='store_true', default=False)
parser.add_argument('--log_attn_weights', action='store_true', default=False)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--mask_prob', type=float, default=0.0)
parser.add_argument('--num_tasks', type=int, default=1)
parser.add_argument('--no', type=int, default=0)
parser.add_argument('--name', type=str, default='')
parser.add_argument('--factor', type=float, default=0.7)  # 0.5
parser.add_argument('--patience', type=int, default=10)  # 3

args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, cos_lr, training_configurations):
    """Sets the learning rate"""

    if not cos_lr:
        if epoch in training_configurations['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations['lr_decay_rate']

    else:
        warm_up = 50
        if epoch < warm_up:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * training_configurations['initial_learning_rate']\
                                    * (1 + math.cos(math.pi * epoch / training_configurations['epochs'])) * epoch / warm_up \
                                    + (1 - epoch / warm_up) * training_configurations['initial_learning_rate'] * 0.1
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * training_configurations['initial_learning_rate']\
                                    * (1 + math.cos(math.pi * epoch / training_configurations['epochs']))


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

device = torch.device("cuda")

t1 = time.time()
train_path = 'StructuredDatasets/train_dataset.pkl'
test_path = 'StructuredDatasets/test2_dataset.pkl'
# processed_path = f"data/{args.dataset}.pt"
processed_path = f"data/{args.dataset}"
# if os.path.exists(processed_path):
train_dataset = torch.load(processed_path + "_train.pt", map_location="cpu")
test_dataset = torch.load(processed_path + "_test.pt", map_location="cpu")
# else:
#     train_data = load_pygdata_3d(train_path)
#     test_data = load_pygdata_3d(test_path)
#     train_dataset = DataListSet(train_data)
#     test_dataset = DataListSet(test_data)
#     torch.save(train_dataset, processed_path + "_train.pt")
#     torch.save(test_dataset, processed_path + "_test.pt")

train_dataset.data = train_dataset.data.to(device)
test_dataset.data = test_dataset.data.to(device)

print('load dataset!', flush=True)
print(f"preprocess {int(time.time()-t1)} s", flush=True)
print(len(train_dataset), len(test_dataset))

loss_fn = nn.L1Loss()
loss_cls = nn.CrossEntropyLoss()
score_fn_MAE = nn.L1Loss()
score_fn_MSE = nn.MSELoss()

record_path = str(args.dataset) + '/' + args.model + args.local_gnn_type + '+' + args.global_model_type + (('_level_' + args.level) if args.model == 'pronet' else '') + (('_equiformer_radius' + str(args.radius) + '_neighbor_' + str(args.max_num_neighbors)) if args.equiformer else '_') + args.name + '/' \
              + 'hid_dim_' + str(args.hid_dim) + '_layers_' + str(args.num_layers) + "_regress_" + str(args.num_layers_regression) + '_cls_' + str(args.num_layers_cls) + '_head_' + str(args.num_heads) + '_pool_' + args.global_pool + '_' + args.act + '_max_length_' + str(args.max_length) \
              + '/mask_' + str(args.mask_prob) + '_lr_' + str(args.lr) + ('_cos-lr_' if args.cos_lr else '') + (('_Plateau_patience_' + str(args.patience) + '_factor_' + str(args.factor)) if not args.cos_lr else '') + 'decay_' + str(args.weight_decay) + \
              '_epochs_' + str(args.epochs) + '_bs_' + str(args.batch_size) + '_drop_' + str(args.dropout) + '_attn_' + str(args.attn_dropout) + '_bn_' + str(args.batch_norm) + '_ln_' + str(args.layer_norm) + '/no_' + str(args.no)
if not os.path.isdir(record_path):
    mkdir_p(record_path)
save_model_name = record_path + '/model.pkl'
save_curve_name = record_path + '/curve.pkl'

accuracy_file = record_path + '/test_MSE_epoch.txt'
record_file = record_path + '/training_process.txt'

print("Save model name:", save_model_name)
print("Save curve name:", save_curve_name)

training_configurations = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'initial_learning_rate': args.lr,
            'changing_lr': [100],
            'lr_decay_rate': 0.5,
            'momentum': 0.9,
            'nesterov': True,
            'weight_decay': args.weight_decay} # 1e-5


def buildMod():
    cfg.attention_type = "block_sparse"
    cfg.block_size = 2
    bigbird_cfg = None if not args.bigbird_cfg else cfg
    if args.model == 'GPS':
        return GPSModel(args.hid_dim, args.out_dim, args.num_layers, args.num_layers_regression, args.num_layers_cls, args.global_pool, args.local_gnn_type,
                    args.global_model_type, args.num_heads, args.act, args.pna_degrees, args.equivstable_pe, args.dropout,
                    args.attn_dropout, args.layer_norm, args.batch_norm, bigbird_cfg, args.log_attn_weights, args.max_length)
    else:
        assert args.level in ['aminoacid', 'backbone', 'allatom']
        return ProNet(args.level, args.num_layers, args.hid_dim, args.out_dim)


def train(mod, opt: AdamW, dl):
    mod.train()
    losss = []
    N = 0
    for batch in dl:
        opt.zero_grad()
        if args.model == 'pronet':
            y = batch.y
            pred = mod(batch)
        else:
            pred, y, logits = mod(batch)
        loss = loss_fn(pred.flatten(), y.flatten())
        loss.backward()
        opt.step()
        num_graphs = batch.num_graphs
        losss.append(loss * num_graphs)
        N += num_graphs
    losss = [_.item() for _ in losss]
    return np.sum(losss) / N


def train_cls(mod, opt: AdamW, dl):
    mod.train()
    losss = []
    N = 0
    for batch in dl:
        mask = torch.bernoulli(torch.full([batch.x.shape[0]], 1-args.mask_prob)).bool()
        label = torch.autograd.Variable(batch.x[mask])
        batch.x[mask] = 26
        opt.zero_grad()
        pred, y, logits = mod(batch)
        # print(label.shape)
        # print(logits[mask].shape)
        loss = loss_cls(logits[mask], label)
        loss.backward()
        opt.step()
        num_graphs = batch.num_graphs
        losss.append(loss * num_graphs)
        N += num_graphs
    losss = [_.item() for _ in losss]
    return np.sum(losss) / N


# def train_hybrid(mod, opt: AdamW, dl):
#     mod.train()
#     losss_reg = []
#     losss_cls = []
#     N = 0
#     for batch in dl:
#         opt.zero_grad()
#         x = torch.autograd.Variable(batch.x)
#         edge_attr = torch.autograd.Variable(batch.edge_attr)
#         pred1, y, logits = mod(batch)
#         loss1 = loss_fn(pred1.flatten(), y.flatten())
#         losss_reg.append(loss1)
#         batch.x = x
#         batch.edge_attr = edge_attr
#         mask = torch.bernoulli(torch.full([batch.x.shape[0]], 1 - args.mask_prob)).bool()
#         label = torch.autograd.Variable(batch.x[mask])
#         batch.x[mask] = 26
#         pred, y, logits = mod(batch)
#         loss2 = loss_cls(logits[mask], label)
#         loss = loss1 + loss2
#         loss.backward()
#         opt.step()
#         num_graphs = batch.num_graphs
#         losss_cls.append(loss2 * num_graphs)
#         N += num_graphs
#     losss_reg = [_.item() for _ in losss_reg]
#     losss_cls = [_.item() for _ in losss_cls]
#     return np.sum(losss_reg) / N, np.sum(losss_cls) / N


@torch.no_grad()
def test(mod, dl):
    mod.eval()
    losss_MAE = []
    losss_MSE = []
    N = 0
    for batch in dl:
        if args.model == 'pronet':
            y = batch.y
            pred = mod(batch)
        else:
            pred, y, logits = mod(batch)
        losss_MAE.append(score_fn_MAE(pred.flatten(), y.flatten()) * batch.num_graphs)
        losss_MSE.append(score_fn_MSE(pred.flatten(), y.flatten()) * batch.num_graphs)
        N += batch.num_graphs
    losss_MAE = [_.item() for _ in losss_MAE]
    losss_MSE = [_.item() for _ in losss_MSE]
    return np.sum(losss_MAE) / N, np.sum(losss_MSE) / N


train_curve = []
test_curve_1 = []
test_curve_2 = []
best_val_score = float("inf")
bs = args.batch_size
trn_dataloader = PygDataloader(train_dataset, batch_size=bs, shuffle=True, drop_last=False)
tst_dataloader = PygDataloader(test_dataset, batch_size=bs)

model = buildMod().to(device)
# model.reset_parameters()
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # lr=3e-4
scd = ReduceLROnPlateau(optimizer, mode="min", factor=args.factor, patience=args.patience)
for i in range(args.epochs):
    if args.cos_lr:
        adjust_learning_rate(optimizer, i, args.cos_lr, training_configurations)
    t1 = time.time()
    # loss = train(model, optimizer, trn_dataloader)
    if args.mask_prob > 0:
        loss_seq = train_cls(model, optimizer, trn_dataloader)
    else:
        loss_seq = 0
    loss_pred = train(model, optimizer, trn_dataloader)
    t2 = time.time()
    # print("train end", flush=True)
    tst_MAE, tst_MSE = test(model, tst_dataloader)
    scd.step(tst_MAE)
    t3 = time.time()
    print(
        f"epoch {i}: train {loss_pred:.4e} {loss_seq:.4e} {int(t2 - t1)}s testMAE {tst_MAE:.4e} testMSE {tst_MSE:.4e} {int(t3 - t2)}s ",
        end="", flush=True)
    # if val_score < best_val_score:
    #     best_val_score = val_score
    #     torch.save(model.state_dict(), save_model_name)
    # t4 = time.time()
    # print(f"tst {tst_score:.4e} {int(t4-t3)}s ", end="")
    # print(optimizer.param_groups[0]['lr'])
    print(flush=True)

    string = str({'TrainMAE': loss_pred, 'TrainCE': loss_seq, 'TestMAE': tst_MAE, 'TestMSE': tst_MSE})
    fd = open(record_file, 'a+')
    fd.write(string + '\n')
    fd.close()

    train_curve.append(loss_pred)
    test_curve_1.append(tst_MAE)
    test_curve_2.append(tst_MSE)

best_epoch_1 = np.argmin(np.array(test_curve_1))
best_epoch_2 = np.argmin(np.array(test_curve_2))
best_train = min(train_curve)

print('Finished training!')
# print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
# print('Test score: {}'.format(test_curve[best_val_epoch]))
print('Best test MAE: {}'.format(np.min(test_curve_1)))
print('Best test MSE: {}'.format(np.min(test_curve_2)))

np.savez(save_curve_name, train=np.array(train_curve), testMAE=np.array(test_curve_1), testMSE=np.array(test_curve_2),
             test_best_1=np.min(test_curve_1), test_best_2=np.min(test_curve_2))
# np.savetxt(accuracy_file, np.array(test_curve_1))

string = 'Best test MAE: ' + str(np.min(test_curve_1))
mean_test_loss = np.mean(np.array(test_curve_1)[-10:])
fd = open(record_file, 'a+')
fd.write(string + '\n')
fd.write('mean test MAE: ' + str(mean_test_loss) + '\n')
fd.close()

string = 'Best test MSE: ' + str(np.min(test_curve_2))
mean_test_loss = np.mean(np.array(test_curve_2)[-10:])
fd = open(record_file, 'a+')
fd.write(string + '\n')
fd.write('mean test MSE: ' + str(mean_test_loss) + '\n')
fd.close()

plt.figure()
plt.plot(test_curve_1, color='b', label='MAE')
plt.plot(test_curve_2, color='r', label='MSE')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.legend()
plt.title('Test Loss')
# plt.show()
plt.savefig(record_path + '/Test_Loss.png')
