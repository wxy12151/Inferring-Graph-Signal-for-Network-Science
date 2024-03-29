# --------------------------
# Importing public libraries
# --------------------------
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import argparse
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, \
    f1_score, roc_auc_score

# --------------------------
# Importing custom libraries
# --------------------------
from utils.preprocess import load_graphs, load_label, get_context_pairs, get_evaluation_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from models.model import DySAT

# --------------------------
# Print to txt file locally
# --------------------------
file_path = './test_logs.txt'
f=open(file_path, 'a')
print('-'*50, file = f)

# --------------------------
# Experimental settings
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', type=int, nargs='?', default=365,
                    help="total time steps used for train, eval and test")
parser.add_argument('--GPU_ID', type=int, nargs='?', default=2,
                    help='GPU_ID (0/1 etc.)')
# parser.add_argument('--epochs', type=int, nargs='?', default=1000,
#                     help='# epochs')
# parser.add_argument('--val_freq', type=int, nargs='?', default=1,
#                     help='Validation frequency (in epochs)')
# parser.add_argument('--test_freq', type=int, nargs='?', default=1,
#                     help='Testing frequency (in epochs)')
parser.add_argument('--batch_size', type=int, nargs='?', default=365,
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
# parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0005, # default = 0.01
#                     help='Initial learning rate for self-attention model.')
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
# Revise it!
# --------------------------
# edge_weight = 'unweighted'
edge_weight = 'hydraulic_loss'
# edge_weight = 'log_hydraulic_loss'
# edge_weight = 'pruned'
# edge_weight = 'pipe_length'
# edge_weight = 'inv_pipe_length'

trained_model_name = 'model'

#----------------------------------------------------------------#
# Load the test dataset
#----------------------------------------------------------------#
graphs_dir = "./data/graphs/graph_2019_{}.pkl".format(edge_weight)
graphs, adjs = load_graphs(graphs_dir ) # 365张图和邻接矩阵，注意点索引是1-782
label_dir = './data/2019_Leakages.csv'
df_label = load_label(label_dir) # 2019 leakage pipes dataset; 105120(365x288) rows × 23(leakages) columns

feats = []
for i in range(len(graphs)):
    feats.append(graphs[i].graph['feature'])

torch.cuda.set_device(args.GPU_ID)
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

dataset = MyDataset(args, graphs, feats, adjs, df_label, label_mode = True)

test_data_size = len(dataset)
print("The length of testing dataset/days is：{}".format(test_data_size), file = f) # 365天的图

dataloader = DataLoader(dataset,  # 定义dataloader # batch_size是512>=365,所以会导入2018年所有图的信息
                        batch_size=args.batch_size, # default 512
                        shuffle=False,
                        num_workers=10, 
                        collate_fn=MyDataset.collate_fn, 
                        drop_last=False # 是否扔掉len % batch_size
                        )

#----------------------------------------------------------------#
# Define Model Structure
#----------------------------------------------------------------#
model = DySAT(args, feats[0].shape[1], args.time_steps).to(device)

#----------------------------------------------------------------#
# Import Trained Model's Parameters - Revise it!
#----------------------------------------------------------------#
model.load_state_dict(torch.load("./model_checkpoints/{}.pt".format(trained_model_name)))

#----------------------------------------------------------------#
# The testing step begins
#----------------------------------------------------------------#
model.eval()
# total_test_loss = 0
total_accuracy = 0
### no grad optimation
with torch.no_grad():
    for idx, feed_dict in enumerate(dataloader): # batch_size是512>365,所以会导入所有节点信息
        feed_dict = to_device(feed_dict, device)
        pyg_graphs, labels = feed_dict.values()
        y_scores = model(pyg_graphs) # list 365 torch.size([782, 2])

        y_score_node = torch.tensor(()).to(device)
        targets = torch.tensor(()).to(device)
        
        for t in range(len(y_scores)): # 遍历每一个时间步骤
            y_score_node = torch.cat((y_score_node, y_scores[t]), 0)
            targets = torch.cat((targets, labels[t].long()))
        
print('The shape of the node scores: {}'.format(y_score_node.shape)) # torch.Size([285430, 2]) 365x782
print('The shape of the node labels: {}'.format(targets.shape)) # torch.Size([285430])

#----------------------------------------------------------------#
# Statistical Metrics
#----------------------------------------------------------------#
    
_, prediction = torch.max(F.softmax(y_score_node, dim = 1), 1)

targets = targets.cpu().numpy()
prediction = prediction.cpu().numpy()

accuracy_count = (prediction == targets).sum()
test_data_size = test_data_size*782 # 782 nodes per day
print("Accuracy:{}".format(accuracy_count/test_data_size), file = f)
# print('micro_precision:{}'.format(precision_score(targets, prediction, average='micro')), file = f)
# print('micro_recall:{}'.format(recall_score(targets, prediction, average='micro')), file = f)
print('recall of 0:{}'.format(recall_score(targets, prediction, pos_label = 0)), file = f)
print('recall of 1:{}'.format(recall_score(targets, prediction, pos_label = 1)), file = f)
# print('micro_f1-score:{}'.format(f1_score(targets, prediction, average='micro')), file = f)
print("Confusion Matrix: ", '\n', confusion_matrix(targets, prediction), file = f)
print("Classification report: ", '\n', classification_report(targets, prediction), file = f)

#----------------------------------------------------------------#
# ROC
#----------------------------------------------------------------#
scores_ = F.softmax(y_score_node, dim = 1).cpu().numpy() # 285430 x 2: 2代表0和1类别的概率

# fpr, tpr, thresholds = roc_curve(targets, scores_[:, 1], pos_label=1) # positive label = 1
# auc_ = auc(fpr, tpr)
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % auc_)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic of ')
# plt.legend(loc="lower right")
# plt.savefig('./figure/ROC/roc_baseline_pipelength.png')
# plt.show()

auc_score =  roc_auc_score(targets, scores_[:, 1])
print("AUC:", auc_score, file = f)

#----------------------------------------------------------------#
# Save predictions for processing to competition results format
#----------------------------------------------------------------#
np.save('./evaluation/targets.npy', targets)
np.save('./evaluation/predictions.npy', prediction)
print("targets and predictions have been saved to folder evaluation", file = f)

f.close()
        


