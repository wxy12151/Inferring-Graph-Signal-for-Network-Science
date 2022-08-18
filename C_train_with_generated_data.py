import os
import argparse
import numpy as np
from numpy import *
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.metrics import recall_score, precision_score
from utils.preprocess import load_graphs, load_label, get_context_pairs, get_evaluation_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from models.model import DySAT
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

from tensorboardX import SummaryWriter

# --------------------------
# Experimental settings
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', type=int, nargs='?', default=365,
                    help="total time steps used for train, eval and test")
parser.add_argument('--GPU_ID', type=int, nargs='?', default=1,
                    help='GPU_ID (0/1 etc.)')
parser.add_argument('--epochs', type=int, nargs='?', default=2000,
                    help='# epochs')
# parser.add_argument('--val_freq', type=int, nargs='?', default=1,
#                     help='Validation frequency (in epochs)')
# parser.add_argument('--test_freq', type=int, nargs='?', default=1,
#                     help='Testing frequency (in epochs)')
parser.add_argument('--batch_size', type=int, nargs='?', default=365,  # every year as a training batch
                    help='Batch size (# nodes)')
# parser.add_argument('--featureless', type=bool, nargs='?', default=True,
#                 help='True if one-hot encoding.')
# parser.add_argument("--early_stop", type=int, default=10,
                    # help="patient")

# --------------------------
# Tunable Hyper-params
# --------------------------
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
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0002, # default = 0.01
                    help='Initial learning rate for self-attention model.')
parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                    help='Spatial (structural) attention Dropout (1 - keep probability).')
parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                    help='Temporal attention Dropout (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                    help='Initial learning rate for self-attention model.')
parser.add_argument('--leakage_weight', type=float, nargs='?', default=100,
                    help='Give leakage labels more weight when getting loss since the biased lables.')

# --------------------------
# Architecture params
# --------------------------
parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,16,8,8,8,8,4,4,4,4,8', # 16,16,8,8,8,8,4,4,4,4,4
                    help='Encoder layer config: # attention heads in each GAT layer')
parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128,128,64,64,64,64,32,32,32,32,64', # 128,128,64,64,64,64,32,32,32,32,32
                    help='Encoder layer config: # units in each GAT layer')
parser.add_argument('--temporal_head_config', type=str, nargs='?', default='8,8,8,8,8', # default = 16
                    help='Encoder layer config: # attention heads in each Temporal layer')
parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='64,64,64,64,64', # default = 128
                    help='Encoder layer config: # units in each Temporal layer')
parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                    help='Position wise feedforward')
parser.add_argument('--window', type=int, nargs='?', default=-1,
                    help='Window for temporal attention (default : -1 => full)')
args = parser.parse_args()
# print(args)

# --------------------------
# Activate GPU and cuda
# --------------------------
torch.cuda.set_device(args.GPU_ID)
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

# --------------------------
# load graphs and labels
# --------------------------
### For training dataset
year_start_ = 2016
year_end_ = 2017
mode = 'real'
# mode = 'nominal'
label_category = 'binary'
# label_category = 'multi'
path_ = './data/generator_data/{}_{}_{}/'.format(year_start_, year_end_, mode)
graphs_train_dir = path_ + "graph_{}_{}_{}.pkl".format(year_start_, year_end_, mode)
graphs_train, adjs_train = load_graphs(graphs_train_dir) # n times 张图和邻接矩阵，注意点索引是1-782
label_train_dir = path_ + 'label_{}_{}_{}.npy'.format(label_category, year_start_, year_end_)
df_label_train = np.load(label_train_dir) # shape: (n_days, 782); num: binary:0/1 or multi:0/1/2.

### For validation dataset in 2018
graphs_valid_dir = "./data/graphs/graph_2018.pkl"
graphs_valid, adjs_valid = load_graphs(graphs_valid_dir) # 365张图和邻接矩阵，注意点索引是1-782
label_valid_dir = './data/2018_Leakages.csv'
df_label_valid = load_label(label_valid_dir) # 2018 leakage pipes dataset; 105120(365x288) rows × 14(leakages) columns

# --------------------------
# 2018 as training dataset and 2019 as valid dataset
# --------------------------

###!!!test!!! also us 2018 as training dataset
# graphs_train_dir = "./data/graphs/graph_2018.pkl"
# graphs_train, adjs_train = load_graphs(graphs_train_dir) # 365张图和邻接矩阵，注意点索引是1-782
# label_train_dir = './data/2018_Leakages.csv'
# df_label_train = load_label(label_train_dir) # 2018 leakage pipes dataset; 105120(365x288) rows × 14(leakages) columns

# ### For validation dataset in 2019
# graphs_valid_dir = "./data/graphs/graph_2019.pkl"
# graphs_valid, adjs_valid = load_graphs(graphs_valid_dir) # 365张图和邻接矩阵，注意点索引是1-782
# label_valid_dir = './data/2019_Leakages.csv'
# df_label_valid = load_label(label_valid_dir) # 2019 leakage pipes dataset; 105120(365x288) rows × 14(leakages) columns

# --------------------------
# Extract nodal features
# --------------------------
# return the list with n_days 782x288 sparse matrices
feats_train = []
for i in range(len(graphs_train)):
    feats_train.append(graphs_train[i].graph['feature'])

feats_valid = []
for i in range(len(graphs_valid)):
    feats_valid.append(graphs_valid[i].graph['feature'])

assert args.time_steps <= len(adjs_train), "Time steps is illegal"

# --------------------------
# Import the dataset
# --------------------------
label_mode_train = False # means df_label does not need the help with self._get_label() in MyDataset class
train_dataset = MyDataset(args, graphs_train, feats_train, adjs_train, df_label_train, label_mode_train)

label_mode_valid = True 
valid_dataset = MyDataset(args, graphs_valid, feats_valid, adjs_valid, df_label_valid, label_mode_valid)
''' 
return:

    self.pyg_graphs: 365-day graph info
    [Data(x=[782, 288], edge_index=[2, 2592], edge_weight=[2592]),
    ......
    Data(x=[782, 288], edge_index=[2, 2592], edge_weight=[2592])]

    self.label: matrix[365 782]--> node is healthy or node in a certrain day 
'''

# --------------------------
# Load the dataset
# --------------------------
train_dataloader = DataLoader(train_dataset,  # 定义dataloader 
                        batch_size=args.batch_size, # default 365
                        shuffle=False,
                        num_workers=10, 
                        collate_fn=MyDataset.collate_fn, 
                        drop_last=False # 是否扔掉len % batch_size
                        )

valid_dataloader = DataLoader(valid_dataset,  # 定义dataloader # batch_size是512>=365,所以会导入2018年所有图的信息
                        batch_size=args.batch_size, # default 365
                        shuffle=False,
                        num_workers=10, 
                        collate_fn=MyDataset.collate_fn, 
                        drop_last=False # 是否扔掉len % batch_size
                        )

# --------------------------
# Define Model and Optimization Strategy
# --------------------------
model = DySAT(args, feats_train[0].shape[1], args.time_steps).to(device) # feats[0].shape: (782, 288)
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# --------------------------
# Add tensorboard
# --------------------------
writer = SummaryWriter("logs")

# --------------------------
# Print traning parameters to txt file locally
# --------------------------
file_path = './test_logs.txt'
f=open(file_path, 'a')
print('*'*80, file = f)
print('learning rate:', args.learning_rate, file = f)
print('epochs:', args.epochs, file = f)
print('structural head config:', args.structural_head_config, file = f)
print('structural layer config:', args.structural_layer_config, file = f)
print('temporal head config:', args.temporal_head_config, file = f)
print('temporal layer config:', args.temporal_layer_config, file = f)
print('leakage weight for getting loss:', args.leakage_weight, file = f)
print('training data generated from {} to {} in {} mode'.format(year_start_, year_end_, mode), file = f)
f.close()

# --------------------------
# Training Start
# --------------------------
start_time = time.time()
best_epoch_loss = 10**9 # infinite
every_n_epoch = 100 # test saved model on 2019 dataset every_n_epoch, print results to .txt file
valid_every_n_epoch = 1 # validate the model every 10 epochs->save time
epoch_loss = [] # store the valid epoch_loss every valid_every_n_epoch epoch for update the model saved
epoch_save = 0 # initialize the epoch that save the model
total_train_step = 0 # batch num, add 1 every batch size
os.environ['MKL_THREADING_LAYER'] = 'GNU'

for epoch in range(args.epochs):
    print("-------Training round {} begins-------".format(epoch+1))
    batch_loss = []
    model.train()
    for idx, feed_dict in enumerate(train_dataloader): # batch_size = 365, 根据collate_fn每次个batch获取一年的graphs和labels进行训练
        feed_dict = to_device(feed_dict, device)
        pyg_graphs, labels = feed_dict.values()
        # pyg_graphs = pyg_graphs.to(device)
        # labels = labels.to(device)
        loss = model.get_loss(pyg_graphs, labels)

        # Optimizer model
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_train_step += 1
        batch_loss.append(loss.item())
        # print("Training Loss on batch {}: {}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss_every_batchsize", loss.item(), total_train_step)

    
    epoch_loss_now = mean(batch_loss)
    print("Training Loss on epoch {}: {}".format(epoch + 1, epoch_loss_now))
    writer.add_scalar("train_loss_every_epoch", epoch_loss_now, epoch + 1)

    # --------------------------
    # Validation Start
    # --------------------------
    if (epoch + 1) % valid_every_n_epoch == 0:
        model.eval()
        ### No grad optimization
        with torch.no_grad():
            for idx, feed_dict in enumerate(valid_dataloader): # batch_size是365 = 365,所以会导入所有节点信息
                feed_dict = to_device(feed_dict, device)
                pyg_graphs, labels = feed_dict.values()
                # forward propagation
                y_scores = model(pyg_graphs) # list 365 torch.size([782, 2])

                # valid loss
                loss = 0
                for t in range(args.time_steps): # 遍历每一个时间步骤
                    emb_t = y_scores[t] #[N, F] 782 2;  获取这一时刻，所有节点的embedding
                    graphloss = model.cirterion(emb_t, labels[t].to(torch.int64))
                    loss += graphloss
                print("Valid Loss on epoch {}: {}".format(epoch + 1, loss.item()))
                writer.add_scalar("valid_loss_every_epoch", loss.item(), epoch + 1)
                epoch_loss.append(loss.item())

                # true labels and predict lables
                y_score_node = torch.tensor(()).to(device)
                targets = torch.tensor(()).to(device)
                for t in range(len(y_scores)): # 遍历每一个时间步骤
                    y_score_node = torch.cat((y_score_node, y_scores[t]), 0)
                    targets = torch.cat((targets, labels[t].long()))
                _, prediction = torch.max(F.softmax(y_score_node, dim = 1), 1)
                targets = targets.cpu().numpy()
                prediction = prediction.cpu().numpy()

                recall_leakage = recall_score(targets, prediction, pos_label = 1)
                print("Valid leakage recall on epoch {}: {}".format(epoch + 1, recall_leakage))
                writer.add_scalar("valid_leakage_recall_every_epoch", recall_leakage, epoch + 1)

                precision_leakage = precision_score(targets, prediction, pos_label = 1)
                print("Valid leakage precision on epoch {}: {}".format(epoch + 1, precision_leakage))
                writer.add_scalar("valid_leakage_precision_every_epoch", precision_leakage, epoch + 1)

        # --------------------------
        # Update the model with the lowest loss
        # --------------------------
        if epoch_loss[-1] < best_epoch_loss:
            best_epoch_loss = epoch_loss[-1]
            epoch_save = epoch + 1
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            print("Update local model on epoch {} with valid loss {}.".format(epoch_save, best_epoch_loss))
    
    # training time every epoch (with valid time)
    end_time = time.time() # one epoch training end
    print("Training Times (with valid time) on epoch {}: {} seconds.".format(epoch + 1, end_time - start_time))

    # --------------------------
    #  Test the saved model on test dataset(2019) every n epochs, output to text_logs.txt
    # --------------------------    
    if (epoch+1) % every_n_epoch == 0:
        file_path = './test_logs.txt'
        f=open(file_path, 'a')
        print('\n', file = f)
        print('-'*50, file = f)
        print('The epoch now is:', epoch+1, file = f)
        print('The tested model now is on epoch {} with loss {}'.format(epoch_save, best_epoch_loss), file = f)
        print('HERE IS THE TEST RESULTS:', file = f)
        f.close()
        os.system("python A_test.py")

    start_time = time.time()
    
file_path = './test_logs.txt'
f=open(file_path, 'a')
print("Finally, the model saved locally is epoch {} with loss {}.".format(epoch_save, best_epoch_loss), file = f)
f.close()

writer.close()
    

    
