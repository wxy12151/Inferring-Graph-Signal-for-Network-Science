# -*- encoding: utf-8 -*-
'''
@File    :   layers.py
@Time    :   2021/02/18 14:30:13
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import softmax
from torch_scatter import scatter

import copy


class StructuralAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, # featurs 288
                output_dim, # output维度 128 64 64
                n_heads, # 多头参数 [16,8,8]-->16
                attn_drop, # drop参数 0.1
                ffd_drop, # 0.1
                residual,
                layer_no): # 残差连接 True
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads  # 每个头特征的维度
        self.n_heads = n_heads
        self.act = nn.ELU()

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)  # 线性层[288, 128]; W*X
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))  # [1, 16, 8]; a1, attention
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))  # [1, 16, 8]; a2, attention
        # att_l初始参数是否需要好的优化
        self.reset_param(self.att_l)
        self.reset_param(self.att_r)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual  # 残差
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)  # [288, 128 = 16*8]
        
        self.layer_no = layer_no

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self, graph):
        # print(graph.x[0])
        if self.layer_no == 1:
            graph = copy.deepcopy(graph) # 注意这里是单时间戳的图
        edge_index = graph.edge_index  # 点边关系 torch.Size([2, 2592])
        edge_weight = graph.edge_weight.reshape(-1, 1) # torch.Size([2592, 1])
        H, C = self.n_heads, self.out_dim # 获取多头信息和每个头特征的维度(128//16)： 16 8

        ### W * x
        # self.lin: Linear(in_features=288, out_features=128, bias=False)
        # graph.x.shape: torch.Size([782, 288])
        # x.shape: torch.Size([782, 16, 8])
        x = self.lin(graph.x).view(-1, H, C) # [N, heads, out_dim]; [782, 288]*[288, 128] => [782,128] => [782,16,8] # 多头attention

        ### attention
        # x.shape: torch.Size([782, 16, 8])
        # self.att_l.shape: torch.Size([1, 16, 8]) # 共享参数a
        # (x * self.att_l).shape: torch.Size([782, 16, 8])
        # ((x * self.att_l).sum(dim=-1)).shape: torch.Size([782, 16])
        # .squeeze: 把所有维度为“1”的压缩
        ## 求得782个点的attention值
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze()  # [N, heads]; a1*X; [782, 16]: 782个节点，16个head，每个head的attention值
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()  # a2*X
        ## 按src和dst列表的顺序分别调取它们的attention值
        alpha_l = alpha_l[edge_index[0]] # [num_edges, heads] 2592 x 16  每个src节点的attention值(16个heads)
        alpha_r = alpha_r[edge_index[1]]  # dst节点特征的attention的值
        ## a^T * [W^s * x_u || W^s * x_v] 2592对src和dst合并矩阵运算 --> alpha: 2592 x 16
        alpha = alpha_r + alpha_l  # 将attention拼接在一起
        ## A_uv * (a^T * [W^s * x_u || W^s * x_v])
        alpha = edge_weight * alpha # [2592, 1] * [2592, 16] -> [2592, 16]
        alpha = self.leaky_relu(alpha) # 经过激活函数
        ## 归一化求得alpha最终表达式
        coefficients = softmax(alpha, edge_index[1]) # [num_edges, heads] 2592 x 16; softmax归一化操作: 先按edge_index[1]进行分组，然后计算softmax值

        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)
        x_j = x[edge_index[0]]  # [num_edges, heads, out_dim] 2592 x 16 x 8 初始节点的特征

        # output; coefficients-每个边对应到的attention系数;
        ## 782个点作为终止节点聚合邻居节点后的最终embedding
        # [nodes, heads, dim] 782 x 16 x 8
        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum"))
        out = out.reshape(-1, self.n_heads*self.out_dim) #[num_nodes, output_dim] 728 x 128
        if self.residual:
            out = out + self.lin_residual(graph.x)  # out加上残差，维度依旧为 728 x 128
        graph.x = out  # 将计算attention后的节点特征赋予到图的节点特征上 728 x128
        # print(graph.x[0])
        return graph

        
class TemporalAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                attn_drop, 
                residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))  # 位置embedding信息[16, 128]
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))  # [128, 128]; W*Q
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """ In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]
            # [N, T, F]; 16个时刻拼接在一起; structural最终输出的节点特征
            # 143 16 128: 143个节点在16个时间步上，通过结构attention层聚合邻居，获得128维度特征输出
        """
        # 1: Add position embeddings to input; [143, 16]: 143个节点，每个节点16个位置信息
        # h_v + p 得到最终输入到self-attention的embedding
        # position_inputs: torch.Size([143, 16]); 重复143个节点; 每个节点有16个时间步 143 x 16;
        position_inputs = torch.arange(0,self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)
        # self.position_embeddings: torch.Size([16, 128])
        # self.position_embeddings[position_inputs]: torch.Size([143, 16, 128])
        temporal_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F] [782, 365, F]; 每个节点在各个时刻对应到的128维向量
        # !!! 以三层structural attention 为例，inputs[349, 0, :]为全0特征，加上position_embedding后temporal_inputs[349, 0, :]有特征了=self.position_embeddings[position_inputs][349, 0, :]

        # 2: Query, Key based multi-head self attention. [143, 16, 128]
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]; 第一个矩阵第2个维度，乘以，第二个矩阵的第0个维度
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)  # 每个head的维度 8
        # 在dim2(特征128)按每个头的特征维度8进行切分，此时有16个[143, 16, 8];接着在dim0进行拼接
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h] [143*16, 16, 8]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T] [143*16, 16, 16]
        outputs = outputs / (self.num_time_steps ** 0.5)  # (Q*K/(d_k**0.5))

        # 4: Masked (causal) softmax to compute attention weights. 目的是将之前没有出现的时间步，设置为0;
        diag_val = torch.ones_like(outputs[0])  # [16,16]的全1向量
        tril = torch.tril(diag_val)  # 下三角阵 后面时间还没有发生，attention程度被认为是0 进行一个mask的操作
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]  重复N次（2288）; [2288, 16, 16]
        padding = torch.ones_like(masks) * (-2**32+1)  # 全为负无穷的 [2288, 16, 16]
        outputs = torch.where(masks==0, padding, outputs)  # outputs中mask为0的地方，填充padding中负无穷的数值
        # 使得做softmax能够给没发生过的时间attention赋值为0
        outputs = F.softmax(outputs, dim=2)  # output:[2288, 16, 16]
        self.attn_wts_all = outputs # [h*N, T, T]
                
        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)  # dropout
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]  # Attention(K*Q)*V; [2288, 16, 8]
        # 还原原来矩阵大小
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F] [143, 16, 128]
        
        # 6: Feedforward and residual
        outputs = self.feedforward(outputs) # 线性层 + relu
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs  # 所有节点聚合时序self-attention后的节点embedding，所有时间

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs # 可认为是残差


    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
