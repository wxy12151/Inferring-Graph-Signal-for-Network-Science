# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/02/19 21:10:00
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss

from models.layers import StructuralAttentionLayer, TemporalAttentionLayer
from utils.utilities import fixed_unigram_candidate_sampler

class DySAT(nn.Module):
    def __init__(self, args, num_features, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DySAT, self).__init__() # 继承自父类nn.Module的init方法
        self.args = args
        if args.window < 0: # 'Window for temporal attention (default : -1 => full)'
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.
        self.num_features = num_features # 143维特征

        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))  # [16, 8, 8] 结构多头信息
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))  # [128, 64, 64] 结构layer层信息
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))  # [16] 时序多头信息
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))  # [128] 时序layer层信息
        self.spatial_drop = args.spatial_drop  # 定义dropout
        self.temporal_drop = args.temporal_drop

        # 定义model
        self.structural_attn, self.temporal_attn = self.build_model()
        self.fc = nn.Linear(self.temporal_layer_config[-1], 2)

        self.bceloss = BCEWithLogitsLoss()  # 定义loss函数; sigmoid和crossentropy组合在一起

        # 标签类别为0, 1，所对应的样本数量为768，14; 对应权重为768/768, 768/14
        self.weights = [1, args.leakage_weight] # default [1, 55]
        self.class_weights = torch.FloatTensor(self.weights)
        self.cirterion = nn.CrossEntropyLoss(weight = self.class_weights) # 定义loss函数; 

    def forward(self, graphs):

        # Structural Attention forward
        structural_out = []
        for t in range(0, self.num_time_steps):  # 遍历每一个时间步的图，节点做类似GAT的操作; 在每一个时间步，输入是节点的特征，输出得到节点聚合邻居后的信息；
            structural_out.append(self.structural_attn(graphs[t]))
        structural_outputs = [g.x[:,None,:] for g in structural_out] # list(365张图) of [N_nodes, 1, out_dim128]/[782, 1, 64]; 节点聚合邻居后的特征

        ## 为在之前时间步里没有出现过的节点特征补0
        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]  # 节点数量: 最后一张图的节点维度782
        out_dim = structural_outputs[-1].shape[-1]  # 上一层输出的点特征维度64

        structural_outputs_padded = []
        for out in structural_outputs:  # 对节点进行补0，使其为同一个维度, 本项目中无用
            # padding节点的数量; 保持一定的维度; torch.Size([125, 1, 128]); 125+18 = 143
            zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)  # 节点特征拼接 [143, 1, 128]
            structural_outputs_padded.append(padded)

        # [N, T, F]; 365个时刻拼接在一起; structural最终输出的节点特征
        # 782 365 64
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1)
        
        # Temporal Attention forward 782个节点在365个时间步上，通过结构attention层聚合邻居，获得64维度特征输出
        temporal_out = self.temporal_attn(structural_outputs_padded)
        # [782 365 64] 782个节点在每一个时间步骤里所对应的节点embedding（经过temp atten 操作后）

        # Fully connected layer
        y_score = []
        for t in range(0, self.num_time_steps): # 遍历每一个时间步的图，全连接输出2维分类
            y_score.append(self.fc(temporal_out[:, t, :])) # list 365 torch.size([782, 2])

        return y_score

    def build_model(self):
        input_dim = self.num_features

        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):  # 结构层信息 range(1) [128] 每一个结构层传递一阶邻居
            layer = StructuralAttentionLayer(input_dim=input_dim,  # featurs 288
                                             output_dim=self.structural_layer_config[i],  # output维度 128 64 64
                                             n_heads=self.structural_head_config[i],  # 多头参数 [16,8,8]-->16
                                             attn_drop=self.spatial_drop,  # drop参数 0.1
                                             ffd_drop=self.spatial_drop, # 0.1
                                             residual=self.args.residual, # 残差连接 True
                                             layer_no = i+1) # the no. of the layer 
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]  # 下一层input维度等于上一层的输出维度
        
        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]  # 时序层信息 输入维度等于结构层最后一层输出维度
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):  # [128] range(1)
            layer = TemporalAttentionLayer(input_dim=input_dim,  # 输入维度
                                           n_heads=self.temporal_head_config[i],  # 多头数量
                                           num_time_steps=self.num_time_steps,  # 时间维度
                                           attn_drop=self.temporal_drop,  # dropout
                                           residual=self.args.residual)  # 残差连接
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers  # 定义的结构层和时序层

    def get_loss(self, pyg_graphs, labels):
        # labels = labels.to(torch.int64) 
        # run gnn
        final_emb = self.forward(pyg_graphs) # list 365 torch.Size([782, 2])
        self.graph_loss = 0
        for t in range(self.num_time_steps): # 遍历每一个时间步骤
            emb_t = final_emb[t] #[N, F] 782 2;  获取这一时刻，所有节点的embedding
            graphloss = self.cirterion(emb_t, labels[t].to(torch.int64))

            # source_node_emb = emb_t[node_1[t]] # [180 128] 初始节点对应的embedding
            # tart_node_pos_emb = emb_t[node_2[t]] # [180 128] 正采样上下文节点对应的embedding
            # tart_node_neg_emb = emb_t[node_2_negative[t]] # [180, 10, 128] 每个正采样节点进行10个负采样
            # pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)  # positive节点内积操作 [180]
            # neg_score = -torch.sum(source_node_emb[:, None, :]*tart_node_neg_emb, dim=2).flatten() # negative节点内积操作 [1800]
            # # self.bceloss: sigmoid和crossentropy组合在一起
            # pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score)) # positive节点和label进行交叉熵 [782, 128] -> 782
            # neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score)) # negtive 节点和label进行交叉熵
            # graphloss = pos_loss + self.args.neg_weight*neg_loss
            self.graph_loss += graphloss
        return self.graph_loss

            




