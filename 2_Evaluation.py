from asyncio import Task, tasks
import numpy as np
import pandas as pd
import epynet
import yaml
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas import DataFrame
import pandas as pd
import networkx as nx

from utils_pre.epanet_loader import get_nx_graph

#----------------------------------------------------------------#
# Revise it!
#----------------------------------------------------------------#
task_name = 'baseline'
# edge_weight = 'pipe_length'
edge_weight = 'inv_pipe_length'
# edge_weight = 'hydraulic_loss'

frequency = 1
distance = 8

#----------------------------------------------------------------#
# import the Grpah of WDN
#----------------------------------------------------------------#
print('1.import the Grpah of WDN')

path_to_wdn = './data/L-TOWN.inp' # Do I need to distinguish between REAL and NOMINAL EPANET inps here? 

# Import the .inp file using the EPYNET library
wdn = epynet.Network(path_to_wdn)

# Solve hydraulic model for a single timestep
wdn.solve()

# Convert the file using a custom function, based on:
# https://github.com/BME-SmartLab/GraphConvWat 
# G: Graph in nx format; pos: node position; head: hydraulic heads which not used in this project
G , pos , head = get_nx_graph(wdn, weight_mode = edge_weight, get_head=True)

#----------------------------------------------------------------#
# Load the Predictions - Revise it!
#----------------------------------------------------------------#
print('2.Load the Predictions')

predictions = np.load('./train_logs/{}_{}/predictions.npy'.format(task_name, edge_weight))
predictions = predictions.reshape((365, 782))

def out_date_by_day(year, day):
    '''
    根据输入的年份和天数计算对应的日期
    '''
    first_day=datetime(year,1,1)
    add_day= timedelta(days=day-1)
    return datetime.strftime(first_day+add_day,"%Y-%m-%d")

#----------------------------------------------------------------#
# Tag Filtering Step 1: Filter by Frequency(>=)
#----------------------------------------------------------------#
print('3.Tag Filtering Step 1: Filter by Frequency')

# frequency = 14

pipe_list = []
date_list = []
for i in range(predictions.shape[1]): # 1-782
    for key in G[i+1]: # 遍历点i的邻居节点->从而遍历管道
        pipe_name = G[i+1][key]['name']
        if len(np.argwhere(predictions[:, i] == 1)) >= frequency: # 如果点i存在1标签 且365天里有>=14天为1
            start_day = np.argwhere(predictions[:, i] == 1)[0] + 1 # 点i泄露开始时间取第一次出现1时
            start_date = out_date_by_day(2019, int(start_day)) # 转为2019年的具体日期
            # print(pipe_name + ', ' + start_date + ' 00:05', file = f)
            pipe_list.append(pipe_name)
            date_list.append(start_date)   

data = {'linkID':pipe_list,
       'startDate':date_list}
df_results = DataFrame(data)
df_results['startDate'] = pd.to_datetime(df_results['startDate'])
df_results.sort_values(by=['startDate', 'linkID'], inplace = True, ignore_index = True) # 根据日期于linkID排序，这里linkID的排序有点小问题，但不影响
dup_row = df_results.duplicated(keep='first') # 找出重复行 keep='first'参数就是让系统从前向后开始筛查，这样索引较大的重复行会返回 'True'。

def get_nodes_of_pipe(G, name):
    '''
    input: G - networkX format graph; name - pipe name, exmaple: 'p253'
    return: two end nodes of the pipe, example: (1, 347)
    '''
    pipe_name_dict = nx.get_edge_attributes(G, 'name')
    for key in pipe_name_dict.keys():
        if pipe_name_dict[key] == name:
            return key

def shortest_length_between_pipes(G, pipe1, pipe2):
    '''
    input: pipe1/pipe2 - two pipes in graph G, example: 'p253'/'p255'
    return: the shortest length between two starting nodes of them, example's length: 2
    '''
    length = None
    end_node_1 = get_nodes_of_pipe(G, pipe1)[0] # 0 for starting node of the pipe
    end_node_2 = get_nodes_of_pipe(G, pipe2)[0]
    if nx.has_path(G, source = end_node_1, target = end_node_2):
        length = nx.shortest_path_length(G, source = end_node_1, target = end_node_2, weight = None)
    return length

#----------------------------------------------------------------#
# Tag Filtering Step 2: Filter by the Distance between Pipes (<=)
#----------------------------------------------------------------#
print('4.Tag Filtering Step 2: Filter by the Distance between Pipes')

# distance = 5

## 方法2: 去除了重复行后，如果pipe1和pipe2在同一天标记为泄漏，且距离<=5，用pipe1代替pipe2形成重复行，再下一单元格继续去除重复行。
# 从而大大减小同一天预测多个管道泄漏，但这些管道离的都很近的问题
# 结果instances从669删减到了141条
# df_results.drop_duplicates(['linkID'], inplace = True, ignore_index = True) # 针对linkID列进行重复删除，这样可能会造成有些在后面日期的正确答案被删除
results_2 = df_results.drop_duplicates(ignore_index = True)
date_list_no_repetition = results_2['startDate'].drop_duplicates()

for date in date_list_no_repetition: # 遍历results_2中每天
    condition1 = results_2['startDate'] == date # 只取在相同date标记为泄漏的pipe
    for pipe1 in results_2[condition1]['linkID']:
        for pipe2 in results_2[condition1]['linkID']:
            length = shortest_length_between_pipes(G, pipe1, pipe2)
            if length and length <= distance: # 如果pipe1和pipe2在同一天标记为泄漏，且距离<=5，用pipe1代替pipe2形成重复行
                condition2 = results_2['linkID'] == pipe2
                index_ = results_2[condition1 & condition2].index.tolist()
                results_2.iloc[index_, 0] = pipe1 # col 0: linkID
results_2 = results_2.drop_duplicates(ignore_index = True)

#----------------------------------------------------------------#
# Save to results_data.txt
#----------------------------------------------------------------#
file_path = './train_logs/{}_{}/results/result_f_{}_distance_{}.txt'.format(task_name, edge_weight, frequency, distance)
print('5.Save to {}.txt'.format(file_path))
f=open(file_path, 'a')
print('# linkID, startTime, endTime, leakDiameter (m), leakType, peakTime', file = f)
for i in range(results_2.shape[0]):
    print(results_2['linkID'][i] + ', ' + results_2['startDate'][i].strftime("%Y-%m-%d") + ' 23:55', file = f)
f.close()