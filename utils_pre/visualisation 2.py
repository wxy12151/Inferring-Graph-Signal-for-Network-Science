#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

+-------------------------------+
|                               |
|   V I S U A L I S A T I O N   | 
|                               |
+-------------------------------+

Created on Tue Jul  6 09:23:33 2021

@author: gardar
"""

import torch
import networkx as nx
import matplotlib.pyplot as plt


# Helper function for visualisation
def visualise(G, pos=None, color='blue', epoch=None, loss=None, axis=None,figsize=(32,32), edge_labels=True, **kwargs):
    
    if axis is None:
        _, axis = plt.subplots(nrows=1,ncols=1, figsize=figsize, dpi = 300)

    plt.xticks([])
    plt.yticks([])
    
    # If coordinates were not passed to the plotting function
    if not pos:
        # Plot as spring layout
        pos = nx.spring_layout(G, seed=42)

    # If this is a trained GNN
    if torch.is_tensor(G):
        G = G.detach().cpu().numpy()
        plt.scatter(G[:, 0], G[:, 1], s=140, c=color, cmap="Set2", **kwargs, ax=axis)
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
            
    # If this is a networkx graph
    else:
        nx.draw_networkx(G, pos=pos, arrows = G.is_directed(),
                         with_labels=True, node_size = 175, 
                         font_size = 7,font_color = 'w',
                         node_color=color, ax=axis)
        #labels = nx.get_edge_attributes(G,'weight')
        
        labels = dict([((u,v,), d['name']) for u,v,d in G.edges(data=True)])
        #labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)])

        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels, ax=axis)
        
    return axis

