import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device




graphs, adjs = load_graphs()

feats = []
for i in range(len(graphs)):
    feats.append(graphs[i].graph['feature'])

time_steps =365
assert time_steps <= len(adjs), "Time steps is illegal"
# node2vec的训练语料; 365个garph 和 365个节点特征;
context_pairs_train = get_context_pairs(graphs, adjs)  # 16个图，每个图中进行随机游走采样;