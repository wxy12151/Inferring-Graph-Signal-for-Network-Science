import os
import argparse
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import epynet 
import torch
from torch.utils.data import DataLoader
import time
import yaml
from scipy.sparse import csr_matrix

from utils.preprocess import load_graphs, load_label, get_context_pairs, get_evaluation_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from utils_pre.epanet_loader import get_nx_graph
from utils_pre.epanet_simulator import epanetSimulator
from utils_pre.data_loader import dataCleaner
from models.model import DySAT

from tensorboardX import SummaryWriter

# --------------------------
# Experimental settings
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', type=int, nargs='?', default=365,
                    help="total time steps used for train, eval and test")
parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                    help='GPU_ID (0/1 etc.)')
parser.add_argument('--epochs', type=int, nargs='?', default=300,
                    help='# epochs')
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
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0001, # default = 0.01
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
parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,16,8', # 16,16,8,8,8,8,4,4,4,4,4
                    help='Encoder layer config: # attention heads in each GAT layer')
parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128,128,64', # 128,128,64,64,64,64,32,32,32,32,32
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

### edge_weight for every graph in 365 days
# edge_weight = 'unweighted'
# edge_weight = 'hydraulic_loss'
# edge_weight = 'log_hydraulic_loss'
# edge_weight = 'pruned'
edge_weight = 'pipe_length'
# edge_weight = 'inv_pipe_length'

### label file direction
label_dir = './data/2018_Leakages.csv'

### test logs file direction
test_logs_dir = './train_logs/reconst_{}/test_logs_with_reconst.txt'.format(edge_weight)

### tensorboard dir
tensor_dir = './train_logs/reconst_{}/logs_with_reconst'.format(edge_weight)

# --------------------------
# Activate GPU and cuda
# --------------------------
torch.cuda.set_device(args.GPU_ID)
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

# --------------------------
# load graphs
# --------------------------

# Runtime configuration
path_to_wdn     = './data/L-TOWN.inp'
path_to_data    = './data/l-town-data/'
scaling         = 'minmax'

# Import the .inp file using the EPYNET library
wdn = epynet.Network(path_to_wdn)

# Solve hydraulic model for a single timestep
wdn.solve()

# Convert the file using a custom function, based on:
# https://github.com/BME-SmartLab/GraphConvWat 
G , pos , head = get_nx_graph(wdn, weight_mode = edge_weight, get_head=True)

# Instantiate the nominal WDN model
nominal_wdn_model = epanetSimulator(path_to_wdn, path_to_data)

# Run a simulation
nominal_wdn_model.simulate()

# Retrieve the nodal pressures
nominal_pressure = nominal_wdn_model.get_simulated_pressure()

# Run a simulation
nominal_wdn_model.simulate()

# Retrieve the nodal pressures
nominal_pressure = nominal_wdn_model.get_simulated_pressure()

# Open the dataset configuration file
with open(path_to_data + 'dataset_configuration.yml') as file:

    # Load the configuration to a dictionary
    config = yaml.load(file, Loader=yaml.FullLoader) 

# Generate a list of integers, indicating the number of the node
# at which a  pressure sensor is present
sensors = [int(string.replace("n", "")) for string in config['pressure_sensors']]

x,y,scale,bias = dataCleaner(pressure_df    = nominal_pressure, # Pass the nodal pressures
                                observed_nodes = sensors,          # Indicate which nodes have sensors
                                rescale        = scaling)          # Perform scaling on the timeseries data

def read_prediction(filename='predictions.csv', scale=1, bias=0, start_date='2018-01-01 00:00:00'):
    df = pd.read_csv(filename, index_col='Unnamed: 0', error_bad_lines=False)
    df.columns = ['n{}'.format(int(node)+1) for node in df.columns]
    df = df*scale+bias
    df.index = pd.date_range(start=start_date,
                             periods=len(df),
                             freq = '5min')
    return df

r18 = read_prediction(filename='./data/reconstruction/2018_reconstructions.csv',
                      scale=scale,
                      bias=bias,
                      start_date='2018-01-01 00:00:00')

begin_year = r18.index[0].date().year
end_year = r18.index[-1].date().year

graphs = []
for i in range((end_year - begin_year + 1)*365): # range(365)
    tmp_feature = []
    for node in G.nodes(): # 1-782
        tmp_feature.append(np.array(r18.iloc[288*i : 288*(i+1), node-1].tolist()))
    G.graph["feature"] = csr_matrix(tmp_feature) 
    graphs.append(G.copy())

adjs = [nx.adjacency_matrix(g) for g in graphs]

# --------------------------
# load labels
# --------------------------

df_label = load_label(label_dir) # 2018 leakage pipes dataset; 105120(365x288) rows × 14(leakages) columns

# --------------------------
# Extract nodal features
# --------------------------
feats = []
for i in range(len(graphs)):
    feats.append(graphs[i].graph['feature'])

assert args.time_steps <= len(adjs), "Time steps is illegal"

# --------------------------
# Import the dataset
# --------------------------
label_mode = True
dataset = MyDataset(args, graphs, feats, adjs, df_label, label_mode)
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
dataloader = DataLoader(dataset,  # 定义dataloader # batch_size是512>=365,所以会导入2018年所有图的信息
                        batch_size=args.batch_size, # default 512
                        shuffle=False,
                        num_workers=10, 
                        collate_fn=MyDataset.collate_fn, 
                        drop_last=False # 是否扔掉len % batch_size
                        )

# --------------------------
# Define Model and Optimization Strategy
# --------------------------
model = DySAT(args, feats[0].shape[1], args.time_steps).to(device) # feats[0].shape: (782, 288)
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# --------------------------
# Add tensorboard
# --------------------------
writer = SummaryWriter(tensor_dir)

# --------------------------
# Print to txt file locally
# --------------------------
file_path = test_logs_dir
f=open(file_path, 'a')
print('*'*80, file = f)
print('learning rate:', args.learning_rate, file = f)
print('epochs:', args.epochs, file = f)
print('structural head config:', args.structural_head_config, file = f)
print('structural layer config:', args.structural_layer_config, file = f)
print('temporal head config:', args.temporal_head_config, file = f)
print('temporal layer config:', args.temporal_layer_config, file = f)
print('leakage weight for getting loss:', args.leakage_weight, file = f)
f.close()

# --------------------------
# Training Start
# --------------------------
start_time = time.time()
best_epoch_loss = float('inf')
every_n_epoch = 50
epoch_loss = []
epoch_save = 0
os.environ['MKL_THREADING_LAYER'] = 'GNU'
for epoch in range(args.epochs):
    model.train()
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

    end_time = time.time()
    print("-"*30)
    print("Training Times on epoch {}: {} seconds.".format(epoch + 1, end_time - start_time))
    print("Training Loss on epoch {}: {}".format(epoch + 1, loss.item()))
    writer.add_scalar("train_loss", loss.item(), epoch + 1)
    
    # Update the model with the lowest loss
    if epoch_loss[-1] < best_epoch_loss:
        best_epoch_loss = epoch_loss[-1]
        epoch_save = epoch + 1
        torch.save(model.state_dict(), "./model_checkpoints/model_reconst_{}.pt".format(edge_weight))
        print("Update local model on epoch {} with loss {}.".format(epoch_save, best_epoch_loss))
    
    # Test the saved model every n epochs
    if (epoch+1) % every_n_epoch == 0:
        file_path = test_logs_dir
        f=open(file_path, 'a')
        print('\n', file = f)
        print('-'*50, file = f)
        print('The epoch now is:', epoch+1, file = f)
        print('The tested model now is on epoch {} with loss {}'.format(epoch_save, best_epoch_loss), file = f)
        print('HERE IS THE TEST RESULTS:', file = f)
        f.close()
        os.system("python B_test_with_reconstructed_data.py")

    start_time = time.time()
    
file_path = test_logs_dir
f=open(file_path, 'a')
print("Finally, the model saved locally is epoch {} with loss {}.".format(epoch_save, epoch_loss[epoch_save-1]), file = f)
f.close()

writer.close()
    

    
