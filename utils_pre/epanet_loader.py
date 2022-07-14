#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

+-------------------------------+
|                               |
|   E P A N E T   L O A D  E R  | 
|                               |
+-------------------------------+

Created on Mon Jul  5 18:29:39 2021

@author: gardar
"""

import numpy as np
import pandas as pd
import networkx as nx
from utils_pre.helpers import list_of_dict_search

""" 
get_nx_graph(wds, mode)

This is a function from the signal nodal pressure reconstruction project from:

    https://github.com/BME-SmartLab/GraphConvWat

The function is used for reading EPANET .inp files using the python library
EPYNET, and converting them into networkx graphs.

-------------------------------------------------------------------------------
Revision history:
-------------------------------------------------------------------------------
The function has been adjusted for this leakage detection study.
Renamed 'mode' to a more descriptive 'weight_mode'

Renamed the graph weight modes:
    binary      -> unweighted
    weighted    -> hydraulic_loss
    logarithmic -> log_hydraulic_loss
    pruned      =  pruned

... and added a new weight mode that uses just the pipe length:
    *           -> pipe_length
    
- Garðar Örn Garðarsson, 5 July 2021
  Vað, Skriðdal 

"""

def get_nx_graph(wds, weight_mode='unweighted', get_head=False):
    
    # Instantiate a dict of junctions
    junc_dict = {}
    
    # There is an option to retrieve the hydraulic head measurements
    if get_head:
        head_dict = {}
    
    # Populate the junction list
    for junction in wds.junctions:
        junc_dict[junction.index] = (junction.coordinates[0], junction.coordinates[1])
        
        # We generate a dictionary of hydraulic head data addressed like the GIS coords
        if get_head:
            head_dict[junction.index] = junction.head
    
    # Then convert the head data to a pandas series for easier manipulation
    if get_head:
        head = pd.Series(head_dict)
    
    # Instantiate an empty graph
    G = nx.Graph()
    
    # Populate the graph
    
    # Binary mode generates an unweighted graph
    if weight_mode == 'unweighted':
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_dict) and (pipe.to_node.index in junc_dict):
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=1., name=str(pipe).replace("<epynet.Pipe with id '","").replace("'>", ""))
        for pump in wds.pumps:
            if (pump.from_node.index in junc_dict) and (pump.to_node.index in junc_dict):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1., name=str(pump).replace("<epynet.Pump with id '","").replace("'>", ""))
        for valve in wds.valves:
            if (valve.from_node.index in junc_dict) and (valve.to_node.index in junc_dict):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1., name=str(valve).replace("<epynet.Valve with id '","").replace("'>", ""))
    
    # Generate a weighted graph based on hydraulic loss calculations
    elif weight_mode == 'hydraulic_loss':
        max_weight = 0
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_dict) and (pipe.to_node.index in junc_dict):
                weight  = ((pipe.diameter*3.281)**4.871 * pipe.roughness**1.852) / (4.727*pipe.length*3.281)
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=weight, name=str(pipe).replace("<epynet.Pipe with id '","").replace("'>", ""))
                if weight > max_weight:
                    max_weight = weight
        for (_,_,d) in G.edges(data=True):
            d['weight'] /= max_weight
        for pump in wds.pumps:
            if (pump.from_node.index in junc_dict) and (pump.to_node.index in junc_dict):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1., name=str(pump).replace("<epynet.Pump with id '","").replace("'>", ""))
        for valve in wds.valves:
            if (valve.from_node.index in junc_dict) and (valve.to_node.index in junc_dict):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1., name=str(valve).replace("<epynet.Valve with id '","").replace("'>", ""))

    # A logarithmically weighted graph
    elif weight_mode == 'log_hydraulic_loss':
        max_weight = 0
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_dict) and (pipe.to_node.index in junc_dict):
                weight  = np.log10(((pipe.diameter*3.281)**4.871 * pipe.roughness**1.852) / (4.727*pipe.length*3.281))
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=float(weight), name=str(pipe).replace("<epynet.Pipe with id '","").replace("'>", ""))
                if weight > max_weight:
                    max_weight = weight
        for (_,_,d) in G.edges(data=True):
            d['weight'] /= max_weight
        for pump in wds.pumps:
            if (pump.from_node.index in junc_dict) and (pump.to_node.index in junc_dict):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1., name=str(pump).replace("<epynet.Pump with id '","").replace("'>", ""))
        for valve in wds.valves:
            if (valve.from_node.index in junc_dict) and (valve.to_node.index in junc_dict):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1., name=str(valve).replace("<epynet.Valve with id '","").replace("'>", ""))
    
    # Pruned, all weights set to zero
    elif weight_mode == 'pruned':
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_dict) and (pipe.to_node.index in junc_dict):
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=0., name=str(pipe).replace("<epynet.Pipe with id '","").replace("'>", ""))
        for pump in wds.pumps:
            if (pump.from_node.index in junc_dict) and (pump.to_node.index in junc_dict):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=0., name=str(pump).replace("<epynet.Pump with id '","").replace("'>", ""))
        for valve in wds.valves:
            if (valve.from_node.index in junc_dict) and (valve.to_node.index in junc_dict):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=0., name=str(valve).replace("<epynet.Valve with id '","").replace("'>", ""))

    # A pipe length weighted graph
    elif weight_mode == 'pipe_length':
        max_weight = 0
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_dict) and (pipe.to_node.index in junc_dict):
                weight  = pipe.length
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=float(weight), name=str(pipe).replace("<epynet.Pipe with id '","").replace("'>", ""))
                if weight > max_weight:
                    max_weight = weight    
        for (_,_,d) in G.edges(data=True):
            d['weight'] /= max_weight
        for pump in wds.pumps:
            if (pump.from_node.index in junc_dict) and (pump.to_node.index in junc_dict):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1., name=str(pump).replace("<epynet.Pump with id '","").replace("'>", ""))
        for valve in wds.valves:
            if (valve.from_node.index in junc_dict) and (valve.to_node.index in junc_dict):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1., name=str(valve).replace("<epynet.Valve with id '","").replace("'>", ""))
      
    # An inverted pipe length weighted graph
    elif weight_mode == 'inv_pipe_length':
        max_weight = 0
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_dict) and (pipe.to_node.index in junc_dict):
                weight  = 1/(pipe.length)
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=float(weight), name=str(pipe).replace("<epynet.Pipe with id '","").replace("'>", ""))
                if weight > max_weight:
                    max_weight = weight
        for (_,_,d) in G.edges(data=True):
            d['weight'] /= max_weight
        for pump in wds.pumps:
            if (pump.from_node.index in junc_dict) and (pump.to_node.index in junc_dict):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1., name=str(pump).replace("<epynet.Pump with id '","").replace("'>", ""))
        for valve in wds.valves:
            if (valve.from_node.index in junc_dict) and (valve.to_node.index in junc_dict):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1., name=str(valve).replace("<epynet.Valve with id '","").replace("'>", ""))
                   
    # So I have an issue with the sorting of the nodes of the imported graph
    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))            
                
    # Return the graph object a junction dictionary which 
    # holds the GIS coordinates (positions) of the nodes
    # and the hydraulic heads at each node
    if get_head:
        return H, junc_dict, head
    
    # If head is not asked for we don't return it
    else:
        return H, junc_dict
