#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

+---------------------------------------+
|                                       |
|    E P A N E T   S I M U L A T O R    | 
|                                       |
+---------------------------------------+

Library for simulating pressure dependent demand and demand-driven hydraulics
Using this class we may use the EPANET nominal model to simulate complete nodal 
pressures for the WDN. 
From the simulations, we may train the GNN model to reconstruct the graph-signal
from a few observations.

Created on Tue Jul 27 13:37:15 2021

@author: gardar
"""

# Water Network Tool for Resilience (WNTR) for simulating and analysing WDNs
import wntr
# Pandas for data handling
import pandas as pd

# Class for simulating EPANET models using WNTR
# This class is adapted from: https://github.com/erytheis/BattLeDIM
class epanetSimulator:
    
    # Initialise class
    def __init__(self, epanet_file_path, data_path):
        
        # Initialise filepath into self
        self.epanet_file_path = epanet_file_path
        
        # Initialise datapath into self, this is where simulation 'turds' are stored
        self.data_path = data_path
        
        # Load the EPANET model at the given path
        self.model = wntr.network.WaterNetworkModel(epanet_file_path)

    # Simulation method
    def simulate(self):
        
        # Instantiate wntr EPANET simulator
        self.sim = wntr.sim.EpanetSimulator(self.model)
        
        # Run the simulation
        self.results = self.sim.run_sim(file_prefix=self.data_path+'epanetSimulatorTemp')
        
        # Pass the results
        return self.results

    # Result getter
    def get_results(self):
        return self.results

    # Pipe roughness getter
    def get_roughness(self):
        self.roughness = {}
        for pipe_name, pipe in self.model.pipes():
            self.roughness[pipe_name] = pipe.roughness
        self.roughness = pd.DataFrame(self.roughness)
        return self.roughness

    # Node type getter
    def get_node_types(self):
        self.node_types = {}
        for node_name, node in self.model.nodes():
            self.node_types[node_name] = node.node_type
        self.node_types = pd.DataFrame(self.node_types)
        return self.node_types

    # Pressure getter
    def get_simulated_pressure(self):
        if 'R1' and 'R2' and 'T1' in self.results.node['pressure'].columns:
            self.pressure = self.results.node['pressure'].drop(columns=['R1','R2','T1'])
        else:
            self.pressure = self.results.node['pressure']
        return self.pressure

    # Demand getter
    def get_nominal_demand(self):
        self.nominal_demand = {}
        for node_name, node in self.model.nodes():
            if node.node_type == 'Junction':
                self.nominal_demand[node_name] = node.base_demand
        return self.nominal_demand
        
if __name__ == '__main__':
        
    base_dir = "/Users/gardar/Documents/UCL/ELEC0054 IMLS Research Project/04 Implementation/03 Hydraulic Simulations/BattLeDIM"

    epanet_file_path = base_dir + '/L-TOWN.inp'
    
    wn_no_leaks = epanetSimulator(epanet_file_path)

    wn_no_leaks.simulate()
    
    pressures = wn_no_leaks.get_simulated_pressure()

