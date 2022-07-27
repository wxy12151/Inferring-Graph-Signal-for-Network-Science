# --------------------------
# Importing public libraries
# --------------------------
import torch
from torch.utils.data import DataLoader
import time
import argparse
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import pandas as pd

# --------------------------
# Importing custom libraries
# --------------------------
from utils.preprocess import load_graphs, load_label, get_context_pairs, get_evaluation_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from models.model import DySAT

#----------------------------------------------------------------#
# Parameters Setting
#----------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', type=int, nargs='?', default=365,
                    help="total time steps used for train, eval and test")
# Experimental settings.
# parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
#                     help='GPU_ID (0/1 etc.)')
parser.add_argument('--epochs', type=int, nargs='?', default=1000,
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
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.001, # default = 0.01
                    help='Initial learning rate for self-attention model.')
parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                    help='Spatial (structural) attention Dropout (1 - keep probability).')
parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                    help='Temporal attention Dropout (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                    help='Initial learning rate for self-attention model.')

# Architecture params
parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,16,8', # 16,16,8,8,8,8,4,4,4,4,4
                    help='Encoder layer config: # attention heads in each GAT layer')
parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128,128,64', # 128,128,64,64,64,64,32,32,32,32,32
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
# print(args)


#----------------------------------------------------------------#
# Load the test dataset
#----------------------------------------------------------------#

graphs_dir = "./data/graphs/graph_2019.pkl"
graphs, adjs = load_graphs(graphs_dir ) # 365张图和邻接矩阵，注意点索引是1-782
label_dir = './data/2019_Leakages.csv'
df_label = load_label(label_dir) # 2019 leakage pipes dataset; 105120(365x288) rows × 23(leakages) columns

feats = []
for i in range(len(graphs)):
    feats.append(graphs[i].graph['feature'])

device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

dataset = MyDataset(args, graphs, feats, adjs, df_label)

dataloader = DataLoader(dataset,  # 定义dataloader # batch_size是512>=365,所以会导入2018年所有图的信息
                        batch_size=args.batch_size, # default 512
                        shuffle=False,
                        num_workers=10, 
                        collate_fn=MyDataset.collate_fn, 
                        drop_last=False # 是否扔掉len % batch_size
                        )

# 导入模型结构
model = DySAT(args, feats[0].shape[1], args.time_steps).to(device) 

# 导入模型参数
model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
model.eval()
emb = model(feed_dict["graphs"])[:, -2, :].detach().cpu().numpy()
val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                    train_edges_neg,
                                                    val_edges_pos, 
                                                    val_edges_neg, 
                                                    test_edges_pos,
                                                    test_edges_neg, 
                                                    emb, 
                                                    emb)
auc_val = val_results["HAD"][1]
auc_test = test_results["HAD"][1]
print("Best Test AUC = {:.3f}".format(auc_test))