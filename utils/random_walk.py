import numpy as np
import networkx as nx
import random

# DISCLAIMER:
# Parts of this code file are derived from
#  https://github.com/aditya-grover/node2vec

'''Random walk sampling code'''

class Graph_RandomWalk():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        # Node2vec 参数 p q
        self.p = p 
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        返回以start_node为起点，长度为walk_length的游走序列
        '''
        G = self.G
        alias_nodes = self.alias_nodes # 节点的转移概率
        alias_edges = self.alias_edges # Node2vec随机游走概率

        walk = [start_node]

        while len(walk) < walk_length:  # 游走长度20
            cur = walk[-1] # 获取当前走到的节点
            cur_nbrs = sorted(G.neighbors(cur))  # 按照节点的邻居节点排序(因为之前定义alias概率都是对于排序后的邻居节点进行依次定义的)进行概率选择;
            if len(cur_nbrs) > 0:
                if len(walk) == 1:  # 初始节点
                    ### alias_nodes[cur][0]：初始节点的alias J; alias_nodes[cur][1]：初始节点的alias q
                    ### 注意alias_draw()返回的是下一步点在当前点排序邻居中的索引
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])  # alias: 采样，概率
                else:  # 非初始节点，进行node2vec随机游走
                    prev = walk[-2]
                    # alias_edges[(prev, cur)]: 对前个点和当前点进行node2vec采样，[0], [1] 分别代表alias里的J和q
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],  # node2vec
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes()) # G_t1: 782 nodes
        for walk_iter in range(num_walks):  # 每个节点遍历/随机游走的次数：10
            random.shuffle(nodes) # 打乱
            for node in nodes:
                # self.node2vec_walk()返回以start_node为起点，长度为walk_length的游走序列
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        
        # 782 * 10个 20步长的随机游走序列
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        计算dst随机游走前往其邻居节点的概率
        '''
        G = self.G # 以G_t1为例 graph with 782 nodes and 905 edges
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):  # dst终止节点的邻居节点 # 终止节点的意思是当前游走到了dst节点，上一步是src
            # 此时游走到dst（三种情况见ppt解析）
            if dst_nbr == src: # src 初始节点
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)  # node2vec向回走概率
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]  # 下一次游走的概率

        return alias_setup(normalized_probs) # 同样对于归一化的概率进行alias采样预备，返回J和q

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G  # G.adj
        is_directed = self.is_directed

        # 注释以第一帧图为例 # G.nodes() 782个nodes
        alias_nodes = {}
        for node in G.nodes():  # 节点到下一个节点的采样概率
            # 遍历每一个点的邻居节点求 点与其每个邻居节点的边权重
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]  # 节点与邻居节点边的权重
            norm_const = sum(unnormalized_probs) # 点与其邻居节点边权重求和
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]  # 归一化操作
            alias_nodes[node] = alias_setup(normalized_probs)  # 对归一化后的概率进行alias采样，目的是降低时间复杂度

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1]) # 添加一条边
        else: # 无向图
            for edge in G.edges():  # node2vec的采样概率
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])  # node2vec随机游走的概率
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])  # 无向图需要进行点点交换的“有向边”计算

        self.alias_nodes = alias_nodes  # 初始节点到下个节点的概率（第一次游走使用）782个node对应的alias的J和q
        self.alias_edges = alias_edges  # 把每条边两个点当作src和dst进行的node2vec采样概率 905*2种(src, dst)对应的alias的J和q

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))  # 1~N采样/随机数
    if np.random.rand() < q[kk]:  # 0～1生成随机数
        return kk
    else:
        return J[kk]
