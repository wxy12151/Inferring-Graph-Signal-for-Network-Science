
import numpy as np
import copy
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from utils.random_walk import Graph_RandomWalk

import torch


"""Random walk-based pair generation."""

def run_random_walks_n2v(graph, adj, num_walks, walk_len):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using 
        the sampling strategy of node2vec (deepwalk)"""
    nx_G = nx.Graph()
    for e in graph.edges():
        # 本来782个点索引是1-782；现改为0-781
        nx_G.add_edge(e[0]-1, e[1]-1)  # 添加节点; 重复只保留一次; 只将有连接的节点保留，没有连接的去掉（因为在构图时，将之前的节点加入了）
    for edge in graph.edges():
        nx_G[edge[0]-1][edge[1]-1]['weight'] = adj[edge[0]-1, edge[1]-1]  # 连接的边的权重; 也就是连接的次数

    # 经过以上操作，这个图包括节点，边以及边的权重
    # 开始定义node2vec随机游走
    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0) # False: 非有向图; node2vec p=q=1.0
    G.preprocess_transition_probs()  # 游走概率计算
    walks = G.simulate_walks(num_walks, walk_len)  # 随机游走 728(点)*10(num_walks)个采样序列，每个20(walk_len)步

    ### 在规定的window_size里获取节点和其上下文节点作为DySAT损失函数的正样本
    WINDOW_SIZE = 10
    pairs = defaultdict(list)
    pairs_cnt = 0
    # 在之前的7820条随机游走序列中，遍历每条序列中的每个点取window为10上下文加入此点的上下文节点list
    for walk in walks: # 782 * 10个 20步长的随机游走序列
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:  # 该节点的window上下文节点，该点在walk序列中前后的十个点
                if nb_word != word:
                    pairs[word].append(nb_word)  # 和本身节点不同，则作为一对训练语料
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs))) # 782 每个点的上下文节点
    print("# sampled pairs: {}".format(pairs_cnt)) # 1950212(每次数量有差异)
    return pairs

def fixed_unigram_candidate_sampler(true_clasees, 
                                    num_true, 
                                    num_sampled, 
                                    unique,  
                                    distortion, 
                                    unigrams):
    # TODO: implementate distortion to unigrams
    assert true_clasees.shape[1] == num_true
    samples = []
    for i in range(true_clasees.shape[0]):  # 遍历正样本节点;
        dist = copy.deepcopy(unigrams)  # 节点的degree
        candidate = list(range(len(dist)))  # self.graphs[t].nodes
        taboo = true_clasees[i].cpu().tolist()  # 这个正样本节点
        for tabo in sorted(taboo, reverse=True):
            candidate.remove(tabo)  # 移除正样本节点
            dist.pop(tabo)  # 该正样本节点degree去除
        sample = np.random.choice(candidate, size=num_sampled, replace=unique, p=dist/np.sum(dist))  # 每个正样本按概率采样10个节点
        samples.append(sample)
    return samples  # 10个正样本，每个正样本采样10个负样本

def to_device(batch, device):
    feed_dict = copy.deepcopy(batch)
    node_1, node_2, node_2_negative, graphs = feed_dict.values()
    # to device
    feed_dict["node_1"] = [x.to(device) for x in node_1]
    feed_dict["node_2"] = [x.to(device) for x in node_2]
    feed_dict["node_2_neg"] = [x.to(device) for x in node_2_negative]
    feed_dict["graphs"] = [g.to(device) for g in graphs]

    return feed_dict


        



