import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
from nets.molecules_graph_regression.load_net import gnn_model
from train.train_molecules_graph_regression import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
from data.data import LoadData
from torch.utils.data import DataLoader

MODEL_NAME='GatedGCN'
net_params={'L': 4, 'hidden_dim': 70, 'out_dim': 70, 'residual': True, 'edge_feat': False, 'readout': 'mean', 'in_feat_dropout': 0.0, 'dropout': 0.0, 'batch_norm': True, 'pos_enc': False, 'device': 'cpu', 'gpu_id': 0, 'batch_size': 128, 'num_atom_type': 28, 'num_bond_type': 4, 'total_param': 105735}
model = gnn_model(MODEL_NAME, net_params)

root_ckpt_dir ='out/molecules_graph_regression/checkpoints/GatedGCN_ZINC_GPU0_11h05m06s_on_Oct_05_2023'
ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
path_check=ckpt_dir + "/epoch_" + '183'+'.pkl'
state_dict=torch.load(path_check)
model.load_state_dict(state_dict)

device = net_params['device']
model.to(device)


dataset = LoadData('ZINC')
testset=dataset.test
test_loader = DataLoader(testset, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset.collate)

_, test_mae = evaluate_network(model, device, test_loader, 0)

print(test_mae)