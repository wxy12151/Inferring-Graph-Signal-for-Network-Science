from typing import DefaultDict
from collections import defaultdict
from torch.functional import Tensor
from torch_geometric.data import Data
from utils.utilities import fixed_unigram_candidate_sampler
import torch
import numpy as np
import torch_geometric as tg
import scipy.sparse as sp

import datetime


import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, args, graphs, features, adjs, df_label, label_mode):
        super(MyDataset, self).__init__()
        self.args = args
        self.graphs = graphs  # 所有时刻的图
        self.df_label = df_label
        self.features = [self._preprocess_features(feat) for feat in features]  # [365, 782, 288] 365个图的特征进行特征归一化的操作
        self.adjs = [self._normalize_graph_gcn(a)  for a  in adjs]  # 邻居矩阵归一化操作
        self.time_steps = args.time_steps # 步长，即图的个数 365
        # self.context_pairs = context_pairs  # 随机游走序列
        # self.max_positive = args.neg_sample_size  # 负采样数量
        self.train_nodes = list(self.graphs[self.time_steps-1].nodes()) # all nodes in the graph. 所有的节点 1-782

        self.min_t = max(self.time_steps - self.args.window - 1, 0) if args.window > 0 else 0  # 最小时间步==0
        self.degs = self.construct_degs()  # 计算每个时间步图中节点的度 [365 782]
        self.pyg_graphs = self._build_pyg_graphs()  # 定义dataloader: PyG格式的图信息 --> 包括点, 稀疏点边关系, 边对应权重
        if label_mode:
            self.label = self._get_label() # 365 x 782 的label矩阵 1: leakage; 0: healthy
        elif not label_mode:
            self.label = self.df_label
        # self.__createitems__()  # 创建训练语料


    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        adj = sp.coo_matrix(adj, dtype=np.float32)  # 稀疏矩阵 (782, 782)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)  # 自连接边 1810 + 782 = 2592
        rowsum = np.array(adj_.sum(1), dtype=np.float32)  # 行求和，D矩阵 维度: 
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)  # D-1/2
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()  # D-1/2*A*D-1/2
        return adj_normalized

    def _preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        features = np.array(features.todense()) # (782, 288)
        rowsum = np.array(features.sum(1))  # 按行求和 (782,) 只有33个观测点有值，其他均为0
        r_inv = np.power(rowsum, -1).flatten()  # 特征和的倒数 (782,) 只有33个观测点有值，其他均为inf
        r_inv[np.isinf(r_inv)] = 0.  # 无穷值赋值为0 (782,)
        r_mat_inv = sp.diags(r_inv)  # 转换成对角矩阵 (782, 782)
        features = r_mat_inv.dot(features)  # 特征归一化,就是每782个点每个点的288维特征和为1; 当然，非观测点的288个特征值依旧都为0
        return features

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        # different from the original implementation
        # degree is counted using multi graph
        degs = []
        for i in range(self.min_t, self.time_steps):
            G = self.graphs[i]  # 每个时间步的图
            deg = []
            for nodeid in G.nodes():
                deg.append(G.degree(nodeid))  # 每个图中节点的度
            degs.append(deg)
        return degs

    def _build_pyg_graphs(self):
        '''
        存储每个图的PyG格式的Data，每一个Data包括
        x: n x h 的tensor向量, n表示节点的个数，h是节点的属性的维度
        edge_index: 节点连接关系会使用稀疏的方式存储
            例如： edge_index = torch.tensor([[0, 346, 1, ..., 340, 341, 781],
                           [0, 0, 1, ..., 781, 781, 781]], dtype=torch.long) 
                注意这里的点是点索引，并不是WDN里的点编号! 
                例如WDN中点1和点347相连；点782和点341/342相连！
        edge_weight：边对应的权重

        self.pyg_graphs: 365-day graph info
        [Data(x=[782, 288], edge_index=[2, 2592], edge_weight=[2592]),
        ......
         Data(x=[782, 288], edge_index=[2, 2592], edge_weight=[2592])]

        '''
        pyg_graphs = []
        for feat, adj in zip(self.features, self.adjs):  # 特征和邻接矩阵
            x = torch.Tensor(feat)  # 特征转换成tensor
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)  # 将稀疏矩阵转换成index和value的形式
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)  # 定义data
            pyg_graphs.append(data)
        return pyg_graphs

    def _get_label(self):
        '''
        return label: matrix[365 782]--> node is healthy or node in a certrain day 
        '''
        label = np.zeros((365, 782))
        begin = datetime.date(2018,1,1)
        end = datetime.date(2018,12,31)
        for i in range((end - begin).days+1): # range(365)
            # print('-'*40)
            # print("This is the No.{} day".format(i+1))
            # print('-'*40)
            for node in self.graphs[i]: # 当前时间步里的782个节点 1-782
                for nbr in sorted(self.graphs[i].neighbors(node)): # 每个节点对应的邻居节点 1-782
                    if self.graphs[i][node][nbr]['name'] in self.df_label.columns.tolist(): # 如果这两个节点构成的边属性'name'在泄露管道列表中
                        # print("leakage pipe {}'s nodes are: {} and {}.".format(graphs[i][node][nbr]['name'], node, nbr))
                        if self.df_label[self.graphs[i][node][nbr]['name']][288*i : 288*(i+1)].max() > 0:
                            label[i, node-1] = 1 # 注意-1
                            label[i, nbr-1] = 1
        return label

    def __len__(self):
        return len(self.pyg_graphs) # 365

    def __getitem__(self, index):
        return self.pyg_graphs[index], self.label[index]  # 时间index对应到的图信息
    
    def __createitems__(self):
        self.data_items = {}
        for node in list(self.graphs[self.time_steps-1].nodes()):  # 所有节点
            feed_dict = {}
            node_1_all_time = []
            node_2_all_time = []
            for t in range(self.min_t, self.time_steps):  # 遍历所有的时间步
                node_1 = []
                node_2 = []
                if len(self.context_pairs[t][node]) > self.max_positive:  # 每个节点随机游走后的上下文训练pair
                    node_1.extend([node]* self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][node], self.max_positive, replace=False))  # 随机选择10个训练语料
                else:  # 如果上线文节点数量小于采样数
                    node_1.extend([node]* len(self.context_pairs[t][node]))  # 不抽样
                    node_2.extend(self.context_pairs[t][node])
                assert len(node_1) == len(node_2)
                node_1_all_time.append(node_1)
                node_2_all_time.append(node_2)

            node_1_list = [torch.LongTensor(node) for node in node_1_all_time]  # 节点在每个时间步中，转换成torch
            node_2_list = [torch.LongTensor(node) for node in node_2_all_time]
            node_2_negative = []  # 负采样
            for t in range(len(node_2_list)):  # 每个时间步，对节点负采样
                degree = self.degs[t]  # 该时刻每个节点的degree
                node_positive = node_2_list[t][:, None]  # 正样本节点;
                node_negative = fixed_unigram_candidate_sampler(true_clasees=node_positive,  # t时刻节点的负采样节点
                                                                num_true=1,
                                                                num_sampled=self.args.neg_sample_size,  # 负采样数量
                                                                unique=False,
                                                                distortion=0.75,
                                                                unigrams=degree)
                node_2_negative.append(node_negative)  # 16个时间步
            node_2_neg_list = [torch.LongTensor(node) for node in node_2_negative]
            feed_dict['node_1']=node_1_list  # 该节点
            feed_dict['node_2']=node_2_list  # 该节点的上下文正样本节点
            feed_dict['node_2_neg']=node_2_neg_list  # 该节点负采样节点
            feed_dict["graphs"] = self.pyg_graphs  # 图的信息
        
            self.data_items[node] = feed_dict  # 节点对应到的所有信息

    @staticmethod
    def collate_fn(samples):  # 时间步对应的图信息; 
        ''' 
        return:

            self.pyg_graphs: 365-day graph info
            [Data(x=[782, 288], edge_index=[2, 2592], edge_weight=[2592]),
            ......
            Data(x=[782, 288], edge_index=[2, 2592], edge_weight=[2592])]

            self.label: matrix[365 782]--> node is healthy or node in a certrain day 
        '''
        batch_dict = {}
        graphs_list = []
        labels_list = []
        for sample in samples: # 365
            # print(len(sample))
            graphs_list.append(sample[0])
            labels_list.append(sample[1])
            
        batch_dict["pyg_graphs"] = graphs_list # 365 x pyg.graph info
        batch_dict["labels"] = labels_list # 365 x 782 0/1 labels
        
        # for key in ["node_1", "node_2", "node_2_neg"]:  # node_1:节点本身; node_2:节点对应到的10个pos节点; node_2_neg:节点对应到的[10,10]的负采样节点
        #     data_list = []
        #     for sample in samples:  # 每一个节点的所有信息
        #         data_list.append(sample[key])
        #     concate = []  # 按照时间步涉及到的节点
        #     for t in range(len(data_list[0])):  # 遍历每个时间步
        #         concate.append(torch.cat([data[t] for data in data_list]))  # 对于所有节点，都选择t这个时间步中的节点信息;
        #     batch_dict[key] = concate  # key下的所有时间涉及到的节点
        # batch_dict["graphs"] = samples[0]["graphs"]  # graph
        return batch_dict  # 每个类别下，所有时间步中涉及到的节点（16个时间步，每个时间步中的节点flatten）


    
