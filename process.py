# --------------------------
# Importing public libraries
# --------------------------

# Operating system specific functions
import os

# Argument parser, for configuring the program execution
import argparse

# An object oriented library for handling EPANET files in Python
import epynet

# yaml / yml configuration file support (a kind of language)
# pip install pyyaml
import yaml

# PyTorch deep learning framework
import torch

# Import the networkx library
import networkx as nx

# Import Pandas for data handling
import pandas as pd

# Import numpy for array handling
import numpy as np

# Matplotlib for generating graphics
import matplotlib.pyplot as plt

# PyTorch from graph conversion tool
from torch_geometric.utils import from_networkx

# conda install pytorch-sparse -c pyg

# Train-test split with shuffle
from sklearn.model_selection import train_test_split

import pandas as pd
import datetime
from scipy.sparse import csr_matrix
import pickle as pkl

# --------------------------
# Importing custom libraries
# --------------------------

# Import a custom tool for converting EPANET .inp files to networkx graphs
from utils.epanet_loader import get_nx_graph

# Function for visualisationa
from utils.visualisation import visualise

# EPANET simulator, used to generate nodal pressures from the nominal model
from utils.epanet_simulator import epanetSimulator

# SCADA timeseries dataloader
from utils.data_loader import battledimLoader, dataCleaner, dataGenerator

# PyTorch early stopping callback
from utils.early_stopping import EarlyStopping

# Metrics
from utils.metrics import Metrics

# --------------------------
# NetworkX Graph Conversion
# --------------------------
# Environment Paths
path_to_wdn = './data/L-TOWN.inp'
# Import the .inp file using the EPYNET library
wdn = epynet.Network(path_to_wdn)
# Solve hydraulic model for a single timestep
wdn.solve()
# Convert the file using a custom function, based on:
# https://github.com/BME-SmartLab/GraphConvWat
# G: Graph in nx format; pos: node position; head: hydraulic heads which not used in this project
G , pos , head = get_nx_graph(wdn, weight_mode='pipe_length', get_head=True)

# --------------------------
# Configuration File Import-->find the nodes with pressure sensor
# --------------------------
'''
"dataset_configuration.yalm" :
Configuration file for generating the Historical (2018) and Evaluation (2019) datasets of the BattLeDIM competition.
Contains:
-file name of network
-dataset start and end times
-leakage information (link ID, start Time, end Time, leak Diameter (m), leak Type, peak Time)
-sensor locations (link or node IDs)
'''
print('Importing dataset configuration...\n')

# Open the dataset configuration file
with open('./data/dataset_configuration.yaml') as file:

    # Load the configuration to a dictionary
    config = yaml.load(file, Loader=yaml.FullLoader)

# Generate a list of integers, indicating the number of the node
# at which a  pressure sensor is present
sensors = [int(string.replace("n", "")) for string in config['pressure_sensors']]

# --------------------------
# Generate df_pressure dataframe for all nodes
# --------------------------
# Load the data into a numpy array with format matching the GraphConvWat problem
pressure_2018 = battledimLoader(observed_nodes = sensors,
                                n_nodes        = 782,
                                path           = './data/',
                                file           = '2018_SCADA_Pressures.csv')

# Print information and instructions about the imported data
msg = "The imported sensor data has shape (i,n,d): {}".format(pressure_2018.shape)

print(msg + "\n" + len(msg)*"-" + "\n")
print("Where: ")
print("'i' is the number of observations: {}".format(pressure_2018.shape[0]))
print("'n' is the number of nodes: {}".format(pressure_2018.shape[1]))
print("'d' is a {}-dimensional vector consisting of the pressure value and a mask ".format(pressure_2018.shape[2]))
print("The mask is set to '1' on observed nodes and '0' otherwise\n")

print("\n" + len(msg)*"-" + "\n")
df_pressure = pd.DataFrame(pressure_2018[:,:,0])
df_pressure_origin = pd.read_csv('./data/2018_SCADA_Pressures.csv', sep=';', decimal=',')
df_pressure['datetime'] = df_pressure_origin['Timestamp']

# --------------------------
# Generate 365 graphs in 2018, add day-series pressure measurements as features
# --------------------------
begin = datetime.date(2018,1,1)
end = datetime.date(2018,12,31)
graphs = []
for i in range((end - begin).days+1): # range(365)
    tmp_feature = []
    for node in G.nodes(): # 1-782
        tmp_feature.append(np.array(df_pressure.iloc[288*i : 288*(i+1), node-1].tolist()))
    G.graph["feature"] = csr_matrix(tmp_feature) # need to copy G
    graphs.append(G.copy())

save_path = './data/graphs/graph.pkl'
with open(save_path, "wb") as f:
    pkl.dump(graphs, f)
print("Processed Data Saved at {}".format(save_path))



