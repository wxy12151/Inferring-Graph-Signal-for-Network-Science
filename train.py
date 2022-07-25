import argparse
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from utils.preprocess import load_graphs, load_label, get_context_pairs, get_evaluation_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from models.model import DySAT
import pandas as pd

import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', type=int, nargs='?', default=365,
                    help="total time steps used for train, eval and test")

# Experimental settings.
# parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
#                     help='GPU_ID (0/1 etc.)')
parser.add_argument('--epochs', type=int, nargs='?', default=200,
                    help='# epochs')
# parser.add_argument('--val_freq', type=int, nargs='?', default=1,
#                     help='Validation frequency (in epochs)')
# parser.add_argument('--test_freq', type=int, nargs='?', default=1,
#                     help='Testing frequency (in epochs)')
parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                    help='Batch size (# nodes)')
# parser.add_argument('--featureless', type=bool, nargs='?', default=True,
#                 help='True if one-hot encoding.')
# parser.add_argument("--early_stop", type=int, default=10,
                    # help="patient")

# 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
# Tunable hyper-params
# TODO: Implementation has not been verified, performance may not be good.
parser.add_argument('--residual', type=bool, nargs='?', default=True,
                    help='Use residual')
# # Number of negative samples per positive pair.
# parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
#                     help='# negative samples per positive')
# # Walk length for random walk sampling.
# parser.add_argument('--walk_len', type=int, nargs='?', default=20,
#                     help='Walk length for random walk sampling')
# # Weight for negative samples in the binary cross-entropy loss function.
# parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
#                     help='Weightage for negative samples')
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                    help='Initial learning rate for self-attention model.')
parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                    help='Spatial (structural) attention Dropout (1 - keep probability).')
parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                    help='Temporal attention Dropout (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                    help='Initial learning rate for self-attention model.')

# Architecture params
parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                    help='Encoder layer config: # attention heads in each GAT layer')
parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128,64,64',
                    help='Encoder layer config: # units in each GAT layer')
parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                    help='Encoder layer config: # attention heads in each Temporal layer')
parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                    help='Encoder layer config: # units in each Temporal layer')
parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                    help='Position wise feedforward')
parser.add_argument('--window', type=int, nargs='?', default=-1,
                    help='Window for temporal attention (default : -1 => full)')
args = parser.parse_args()
print(args)

graphs_dir = "./data/graphs/graph.pkl"
graphs, adjs = load_graphs(graphs_dir ) # 365张图和邻接矩阵，注意点索引是1-782
label_dir = './data/2018_Leakages.csv'
df_label = load_label(label_dir) # 2018 leakage pipes dataset; 105120(365x288) rows × 14(leakages) columns

feats = []
for i in range(len(graphs)):
    feats.append(graphs[i].graph['feature'])

time_steps =365
assert time_steps <= len(adjs), "Time steps is illegal"

# node2vec的训练语料; 在365个时间步的图都获得了每个节点对应的上下文节点(随机游走获得)
# context_pairs_train = get_context_pairs(graphs, adjs)  # 365个图，每个图中进行随机游走采样;

# build dataloader and model
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')


dataset = MyDataset(args, graphs, feats, adjs, df_label)
''' 
return:

    self.pyg_graphs: 365-day graph info
    [Data(x=[782, 288], edge_index=[2, 2592], edge_weight=[2592]),
    ......
    Data(x=[782, 288], edge_index=[2, 2592], edge_weight=[2592])]

    self.label: matrix[365 782]--> node is healthy or node in a certrain day 
'''

dataloader = DataLoader(dataset,  # 定义dataloader # batch_size是512>=365,所以会导入2018年所有图的信息
                        batch_size=args.batch_size, # default 512
                        shuffle=True,
                        num_workers=10, 
                        collate_fn=MyDataset.collate_fn, 
                        drop_last=False # 是否扔掉len % batch_size
                        )

model = DySAT(args, feats[0].shape[1], args.time_steps).to(device) # feats[0].shape: (782, 288)
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

### Training Start

for epoch in range(args.epochs):
    model.train()
    epoch_loss = []
    for idx, feed_dict in enumerate(dataloader): # batch_size是512>365,所以会导入所有节点信息
        feed_dict = to_device(feed_dict, device)
        pyg_graphs, labels = feed_dict.values()
        # pyg_graphs = pyg_graphs.to(device)
        # labels = labels.to(device)
        opt.zero_grad()
        loss = model.get_loss(pyg_graphs, labels)
        loss.backward()
        opt.step()
        epoch_loss.append(loss.item())
