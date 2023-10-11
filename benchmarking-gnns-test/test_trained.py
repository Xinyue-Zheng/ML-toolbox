import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
from data.data import LoadData # import dataset



def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device




def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details",
                        default='configs/molecules_graph_regression_GAT_ZINC_100k.json')
    parser.add_argument('--gpu_id', help="Please give a value for gpu id", default=None)
    parser.add_argument('--model', help="Please give a value for model name", default='GAT')
    parser.add_argument('--dataset', help="Please give a value for dataset name", default='ZINC')
    parser.add_argument('--ckpt_path', help="the root for model to load", default='out/molecules_graph_regression/checkpoints/GAT_ZINC_GPU0_11h31m26s_on_Oct_11_2023/RUN_/epoch_153.pkl')
    parser.add_argument('--out_dir', help="Please give a value for out_dir", default=None)
    parser.add_argument('--seed', help="Please give a value for seed", default=41)
    parser.add_argument('--epochs', help="Please give a value for epochs", default=None)
    parser.add_argument('--batch_size', help="Please give a value for batch_size", default=None)
    parser.add_argument('--init_lr', help="Please give a value for init_lr", default=None)
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor", default=None)
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience", default=None)
    parser.add_argument('--min_lr', help="Please give a value for min_lr", default=None)
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay", default=None)
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval", default=None)
    parser.add_argument('--L', help="Please give a value for L", default=None)
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim", default=None)
    parser.add_argument('--out_dim', help="Please give a value for out_dim", default=None)
    parser.add_argument('--residual', help="Please give a value for residual", default=None)
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat", default=None)
    parser.add_argument('--readout', help="Please give a value for readout", default=None)
    parser.add_argument('--kernel', help="Please give a value for kernel", default=None)
    parser.add_argument('--n_heads', help="Please give a value for n_heads", default=None)
    parser.add_argument('--gated', help="Please give a value for gated", default=None)
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout", default=None)
    parser.add_argument('--dropout', help="Please give a value for dropout", default=None)
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm", default=None)
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm", default=None)
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator", default=None)
    parser.add_argument('--data_mode', help="Please give a value for data_mode", default=None)
    parser.add_argument('--num_pool', help="Please give a value for num_pool", default=None)
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block", default=None)
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim", default=None)
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio", default=None)
    parser.add_argument('--linkpred', help="Please give a value for linkpred", default=None)
    parser.add_argument('--cat', help="Please give a value for cat", default=None)
    parser.add_argument('--self_loop', help="Please give a value for self_loop", default=None)
    parser.add_argument('--max_time', help="Please give a value for max_time", default=None)

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']

    if args.batch_size is not None:
        net_params['batch_size']=args.batch_size
    else:
        net_params['batch_size'] = config['params']['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated == 'True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred == 'True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat == 'True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False

    # Superpixels(need to be changed when the method changed)
    # from nets.superpixels_graph_classification.load_net import gnn_model  # import all GNNS
    # testset = dataset.test
    # net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    # net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    # num_classes = len(np.unique(np.array(dataset.train[:][1])))
    # net_params['n_classes'] = num_classes
    #
    # drop_last = True if MODEL_NAME == 'DiffPool' else False
    #
    # if MODEL_NAME in ['RingGNN', '3WLGNN']:
    #     # import train functions specific for WL-GNNs
    #     from train.train_superpixels_graph_classification import train_epoch_dense as train_epoch, \
    #         evaluate_network_dense as evaluate_network
    #     test_loader = DataLoader(testset, shuffle=False, collate_fn=dataset.collate_dense_gnn)
    #
    # else:
    #     # import train functions for all other GCNs
    #     from train.train_superpixels_graph_classification import train_epoch_sparse as train_epoch, \
    #         evaluate_network_sparse as evaluate_network
    #
    #     test_loader = DataLoader(testset, batch_size=net_params['batch_size'], shuffle=False, drop_last=drop_last,
    #                              collate_fn=dataset.collate)


    #ZINC graph regression, need to be changed if the task is different
    from nets.molecules_graph_regression.load_net import gnn_model  # import all GNNS
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    testset = dataset.test

    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        # import train functions specific for WL-GNNs
        from train.train_molecules_graph_regression import evaluate_network_dense as evaluate_network
        test_loader = DataLoader(testset, shuffle=False, collate_fn=dataset.collate_dense_gnn)

    else:
        # import train functions for all other GCNs
        from train.train_molecules_graph_regression import evaluate_network_sparse as evaluate_network
        test_loader = DataLoader(testset, batch_size=net_params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type

    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        num_nodes = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
        max_num_node = max(num_nodes)
        net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']

    if MODEL_NAME == 'RingGNN':
        num_nodes = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
    #load the trained model
    ckpt_path=args.ckpt_path
    if not os.path.exists(ckpt_path):
        raise SystemExit('there is no checkpoint')
    model = gnn_model(MODEL_NAME, net_params)
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    model.to(device)

    # output th test result

    _, test_result = evaluate_network(model, device, test_loader, 0)
    print('test mae:', format(test_result))

main()