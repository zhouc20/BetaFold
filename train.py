import time
import numpy as np
import torch.nn as nn
import torch
from data.datalist import DataListSet
from data.pyg_data_loader import load_pyg_data
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader as PygDataloader
import argparse
import random
import os, sys
import math
import errno
import matplotlib.pyplot as plt
from layer.GPS_model import GPSModel


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
parser.add_argument("--dataset", type=str, default="ProTstab")
parser.add_argument('--model', type=str, default='Transformer')
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
parser.add_argument('--dropout', type=float, default=0.0, help='drop out rate')
parser.add_argument('--attn_dropout', type=float, default=0.0)
parser.add_argument('--batch_norm', action='store_true', default=False)
parser.add_argument('--layer_norm', action='store_true', default=False)
parser.add_argument('--bigbird_cfg', default=None)
parser.add_argument('--log_attn_weights', action='store_true', default=False)
parser.add_argument('--max_length', type=int, default=512)
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
# processed_path = f"data/{args.dataset}.pt"
processed_path = f"data/{args.dataset}"
if os.path.exists(processed_path):
    train_dataset = torch.load(processed_path + "_train.pt", map_location="cpu")
    test_dataset = torch.load(processed_path + "_test.pt", map_location="cpu")
    valid_dataset = torch.load(processed_path + "_valid.pt", map_location="cpu")
else:
    train_data, valid_data, test_data = load_pyg_data()
    train_dataset = DataListSet(train_data)
    test_dataset = DataListSet(test_data)
    valid_dataset = DataListSet(valid_data)
    torch.save(train_dataset, processed_path + "_train.pt")
    torch.save(test_dataset, processed_path + "_test.pt")
    torch.save(valid_dataset, processed_path + "_valid.pt")

train_dataset.data = train_dataset.data.to(device)
test_dataset.data = test_dataset.data.to(device)
valid_dataset.data = valid_dataset.data.to(device)

print('load dataset!', flush=True)
print(f"preprocess {int(time.time()-t1)} s", flush=True)
print(len(train_dataset), len(test_dataset), len(valid_dataset))

loss_fn = nn.L1Loss()
score_fn = nn.L1Loss()

record_path = str(args.dataset) + '_' + args.local_gnn_type + '+' + args.global_model_type + '/' + args.name + '_' + '/' \
              + 'hid_dim_' + str(args.hid_dim) + '_layers_' + str(args.num_layers) + "_regress_" + str(args.num_layers_regression) + '_head_' + str(args.num_heads) + '_pool_' + args.global_pool + args.act + '_max_length_' + str(args.max_length) \
              + '/lr_' + str(args.lr) + ('_cos-lr_' if args.cos_lr else '') + (('_Plateau_patience_' + str(args.patience) + '_factor_' + str(args.factor)) if not args.cos_lr else '') + '_decay_' + str(args.weight_decay) + \
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
    return GPSModel(args.hid_dim, args.out_dim, args.num_layers, args.num_layers_regression, args.global_pool, args.local_gnn_type,
                    args.global_model_type, args.num_heads, args.act, args.pna_degrees, args.equivstable_pe, args.dropout,
                    args.attn_dropout, args.layer_norm, args.batch_norm, args.bigbird_cfg, args.log_attn_weights, args.max_length)


def train(mod, opt: AdamW, dl):
    mod.train()
    losss = []
    N = 0
    for batch in dl:
        opt.zero_grad()
        pred, y = mod(batch)
        loss = loss_fn(pred.flatten(), y.flatten())
        loss.backward()
        opt.step()
        num_graphs = batch.num_graphs
        losss.append(loss * num_graphs)
        N += num_graphs
    losss = [_.item() for _ in losss]
    return np.sum(losss) / N


@torch.no_grad()
def test(mod, dl):
    mod.eval()
    losss = []
    N = 0
    for batch in dl:
        pred, y = mod(batch)
        losss.append(score_fn(pred.flatten(), y.flatten()) * batch.num_graphs)
        N += batch.num_graphs
    losss = [_.item() for _ in losss]
    return np.sum(losss) / N


train_curve = []
valid_curve = []
test_curve = []
best_val_score = float("inf")
bs = args.batch_size
trn_dataloader = PygDataloader(train_dataset, batch_size=bs, shuffle=True, drop_last=False)
val_dataloader = PygDataloader(valid_dataset, batch_size=bs)
tst_dataloader = PygDataloader(test_dataset, batch_size=bs)

model = buildMod().to(device)
# model.reset_parameters()
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # lr=3e-4
scd = ReduceLROnPlateau(optimizer, mode="min", factor=args.factor, patience=args.patience)
for i in range(args.epochs):
    if args.cos_lr:
        adjust_learning_rate(optimizer, i, args.cos_lr, training_configurations)
    t1 = time.time()
    loss = train(model, optimizer, trn_dataloader)
    t2 = time.time()
    # print("train end", flush=True)
    val_score = test(model, val_dataloader)
    scd.step(val_score)
    t3 = time.time()
    print(
        f"epoch {i}: train {loss:.4e} {int(t2 - t1)}s valid {val_score:.4e} {int(t3 - t2)}s ",
        end="", flush=True)
    if val_score < best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), save_model_name)
    tst_score = test(model, tst_dataloader)
    t4 = time.time()
    print(f"tst {tst_score:.4e} {int(t4-t3)}s ", end="")
    print(optimizer.param_groups[0]['lr'])
    # print(flush=True)

    string = str({'Train': loss, 'Validation': val_score, 'Test': tst_score})
    fd = open(record_file, 'a+')
    fd.write(string + '\n')
    fd.close()

    train_curve.append(loss)
    valid_curve.append(val_score)
    test_curve.append(tst_score)

best_val_epoch = np.argmin(np.array(valid_curve))
best_train = min(train_curve)

print('Finished training!')
print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
print('Test score: {}'.format(test_curve[best_val_epoch]))
print('Best test score: {}'.format(np.min(test_curve)))

np.savez(save_curve_name, train=np.array(train_curve), val=np.array(valid_curve), test=np.array(test_curve),
             test_for_best_val=test_curve[best_val_epoch])
np.savetxt(accuracy_file, np.array(test_curve))

string = 'Best validation score: ' + str(valid_curve[best_val_epoch]) + ' Test score: ' + str(test_curve[best_val_epoch])
mean_test_loss = np.mean(np.array(test_curve)[-10:-1])
fd = open(record_file, 'a+')
fd.write(string + '\n')
fd.write('mean test loss: ' + str(mean_test_loss) + '\n')
fd.close()

plt.figure()
plt.plot(test_curve, color='b')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Test MAE')
# plt.show()
plt.savefig(record_path + '/Test_MAE.png')