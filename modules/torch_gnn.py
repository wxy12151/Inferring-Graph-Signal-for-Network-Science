#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 11:49:15 2021

@author: gardar
"""

# --------------------------
# Importing public libraries
# --------------------------

# Operating system specific functions
import os

# Timing functionality for training
import time

# PyTorch deep learning framework
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

# Import Pandas for data handling
import pandas as pd

# Import numpy for array handling
import numpy as np

# --------------------------
# Importing custom libraries
# --------------------------

# To make sure we don't raise an error on importing project specific 
# libraries, we retrieve the path of the program file ...
filepath = os.path.dirname(os.path.realpath(__file__))

# ... set that as our working directory ...
os.chdir(filepath)

# ... and hop back one level!
os.chdir('..')

# PyTorch early stopping callback
from utils.early_stopping import EarlyStopping

# Metrics
from utils.metrics import Metrics

# Import datahandler for predictions
from utils.data_loader import embedSignalOnGraph


class _GNNbase(torch.nn.Module):
    
    def __init__(self, name, data_generator, device, data_scale, data_bias):
        
        super(_GNNbase, self).__init__()
        
        # Initialise basic parameters
        self.name            = name
        self.device          = device
        self.data_generator  = data_generator
        
        # Training loss
        self.trn_loss        = 0
        
        # Validation metrics calculation, 
        self.data_scale = data_scale
        self.data_bias  = data_bias
        self.metrics    = Metrics(self.data_bias, self.data_scale, self.device)
        
        # Validation loss and metrics
        self.val_loss        = 0
        self.val_rel_err     = 0
        self.val_rel_err_obs = 0
        self.val_rel_err_hid = 0
                
        # A Pandas DataFrame containing the per-epoch results of the model
        self.results = pd.DataFrame(columns=['trn_loss', 
                                             'val_loss', 
                                             'val_rel_err', 
                                             'val_rel_err_o', 
                                             'val_rel_err_h'])
        
        # The number of epochs the model has trained is always the same as the lenght of the results dataframe
        self.epoch   = len(self.results)
        
        # Initialise the best validation loss
        self.best_val_loss = np.inf
        
        # Initialise a first run flag for the training
        self.first_training_run = True
        
        # A header for printing per-epoch statistics during training
        self.header1  = '{:^5}'.format('epoch')
        self.header2  = ''.join(['{:^14}'.format(name) for name in self.results.columns])
        self.header3  = '{:^8}'.format('run_time')
        
        self.header   = self.header1 + self.header2 + self.header3
        
        
    def train_one_epoch(self, optimizer):
        '''
        Train the GNN model for a single epoch

        Parameters
        ----------
        optimizer : torch.optimizer
            Pass the trainer an optimizer object, e.g. ADAM.
        current_epoch : TYPE
            Pass the trainer the current epoch for printing 
            training status purposes.

        Returns
        -------
        None. 
        Training losses and model weights are updated in the object.

        '''
        
        # Every time we start a new training run
        if self.first_training_run:
            print(self.header)              # We print out a header for the session statistics
            self.first_training_run = False # And reset the first run flag
        
        # Set the model to train mode
        self.train()
        
        # Initialise a loss variable to be summed
        tot_loss = 0
        
        # Number of variables in dataset
        n = len(self.data_generator.dataset)
        
        # Let the model class know the current epoch to update stats
        self.epoch += 1
        
        # Iterate over batches
        for batch in self.data_generator:
            
            batch = batch.to(self.device)   # Push batch to computation unit
            optimizer.zero_grad()           # Zero gradients
            out   = self(batch)             # Predict on one batch
            
            loss  = torch.nn.functional.mse_loss(out, batch.y) # Calc. loss
            loss.backward()                                     # Backprop
            optimizer.step()                                    # Parameter update
            
            # Sum total loss
            tot_loss += loss.item() * batch.num_graphs   
            
            # Avg. the loss over the dataset size and update the training loss member
            self.trn_loss = tot_loss / n
        
    
    
    def validate(self):
        '''
        Validate the model on the validation batch

        Returns
        -------
        None. 
        Validation losses and errors are updated in the model object.

        '''
        
        # Set the model to evaluation mode
        self.eval()
        
        # Initialise summable loss and error variables
        tot_loss        = 0
        tot_rel_err     = 0
        tot_rel_err_obs = 0
        tot_rel_err_hid = 0 
        
        # Number of variables in dataset
        n = len(self.data_generator.dataset)
        
        # Iterate over batches
        for batch in self.data_generator:
            
            batch       = batch.to(self.device) # Push batch to computation unit
            out         = self(batch)           # Predict on one batch
            
            loss        = torch.nn.functional.mse_loss(out, batch.y) # Calc. loss
            rel_err     = self.metrics.rel_err(out, batch.y)         # Calc. relative err.
            rel_err_obs = self.metrics.rel_err(out,                  # Calc. observed node relative err.
                                          batch.y, 
                                          batch.x[:, -1].type(torch.bool))
            rel_err_hid = self.metrics.rel_err(out,                  # Calc. hidden node relative err
                                          batch.y,
                                          ~batch.x[:, -1].type(torch.bool)) 
            tot_loss        += loss.item() * batch.num_graphs        # Sum total loss
            tot_rel_err     += rel_err.item() * batch.num_graphs     # Sum relative error
            tot_rel_err_obs += rel_err_obs.item() * batch.num_graphs # Sum observed relative error
            tot_rel_err_hid += rel_err_hid.item() * batch.num_graphs # Sum hidden relative error
            
        self.val_loss        = tot_loss / n         # Update validation loss member 
        self.val_rel_err     = tot_rel_err / n      # Update relative error member
        self.val_rel_err_obs = tot_rel_err_obs / n  # Update observed rel. err. member
        self.val_rel_err_hid = tot_rel_err_hid / n  # Update hidden rel. err. member
        
        
    def predict(self, G, partial_graph_signal):
        '''
        Predict a single passed partially observed graph signal

        Parameters
        ----------
        G : networkx graph
            The graph.
        partial_graph_signal : TYPE
            The partially observed signal.

        Returns
        -------
        pred_graph_signal : TYPE
            A predicted, complete graph signal.

        '''
        n_nodes           = partial_graph_signal.shape[0]                       # Count the number of nodes in the graph for reshaping later
        gnn_input         = embedSignalOnGraph(G, partial_graph_signal)         # Generate a GNN input by embedding timeseries on graph
        gnn_input         = gnn_input.to(self.device)
        pred_graph_signal = self(gnn_input).to('cpu')                           # Make a prediction and return a numpy array of same shape as the one passed
        pred_graph_signal = pred_graph_signal.detach().numpy().reshape(n_nodes,)
        
        return pred_graph_signal
    
    
    def update_results(self):
        '''
        Update the self-contained training and validation results

        Returns
        -------
        None.

        '''
        # A method for updating the Pandas DataFrame containing training and validation results   
        self.latest_results = pd.Series({'trn_loss'      : self.trn_loss,
                                         'val_loss'      : self.val_loss,
                                         'val_rel_err'   : self.val_rel_err,
                                         'val_rel_err_o' : self.val_rel_err_obs,
                                         'val_rel_err_h' : self.val_rel_err_hid})         
        
        self.results = self.results.append(self.latest_results, ignore_index=True)
        
        
    def load_results(self, path_to_logs):
        '''
        Load previous training results

        Parameters
        ----------
        path_to_logs : str
            Path to the last version of the model.

        Returns
        -------
        None.
        Updates self 

        '''
        self.results       = pd.read_csv(path_to_logs)        # Load the results to the self-contained Pandas DataFrame
        self.epoch         = len(self.results)                # Update the number of epochs the model has trained
        self.best_val_loss = self.results['val_loss'].min()   # Loading best validation loss
        
        # Print some info about the results loaded
        print('\n \
               Loaded previous model results...\n \
               --------------------------------------------------\n \
               Model has been trained for:\t{} epochs\n \
               Best validation loss:      \t{} \n \
               Occurred in training round:\t{} '.format(self.epoch, self.best_val_loss, self.results['val_loss'].idxmin()))
        
    def load_model(self, path_to_model, path_to_logs):
        '''
        Load a saved state dictionary of the model

        Parameters
        ----------
        path_to_model : str
            Path to a previous model.
        path_to_logs : str
            Path to previous results.

        Returns
        -------
        None.
        Updates the model weights and self-contained stats

        '''
        # Since we're loading models across compute architectures (gpu/cpu) we need to do some mapping
        # We assume all models are either saved from GPU and loaded on CPU, or trained on GPU and loaded on GPU
        # We don't account for the special case where a model is saved from a CPU and loaded to a GPU, so:
        # (... hang on, what about CPU->CPU, will this work?)
        if self.device == torch.device('cpu'):                                        # If we're running a CPU
            self.load_state_dict(torch.load(path_to_model, map_location=self.device)) # We map GPU->CPU
        else:                                                                         # If we're running a GPU
            self.load_state_dict(torch.load(path_to_model))                           # We just load directly
        self.load_results(path_to_logs)                                               # Load the results from previous training and update self stats
        
    def print_stats(self, epoch_time):
        '''
        Print out the training run statistics

        Parameters
        ----------
        epoch_time : float
            The current execution time for the training run.

        Returns
        -------
        None.
        Prints a message.

        '''
        # Every 20th epoch, we print a header for the stats
        if not self.epoch % 20:
            print(self.header)
        
        # Format the print message field
        epoch    = '{:^5}'.format(self.epoch)
        results  = ''.join(['{:^14.6f}'.format(result) for result in self.latest_results.values])
        run_time = '{:^8.2f}sec'.format(epoch_time)
    
        # Print the statistics
        print(epoch + results + run_time)




class ChebNet(_GNNbase):
    
    def __init__(self, name, data_generator, device, in_channels, out_channels, data_scale, data_bias):
        
        super(ChebNet, self).__init__(name, data_generator, device, data_scale, data_bias)
        
        self.conv1 = ChebConv(in_channels, 120, K=240)
        self.conv2 = ChebConv(120, 60, K=120)
        self.conv3 = ChebConv(60, 30, K=20)
        self.conv4 = ChebConv(30, out_channels, K=1, bias=False)
        
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)
        
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)
        
        torch.nn.init.xavier_normal_(self.conv4.weight)
        
        
    def forward(self, data):
        
        x, edge_index, edge_weight  = data.x, data.edge_index, data.weight
        
        x = F.silu(self.conv1(x, edge_index, edge_weight))
        x = F.silu(self.conv2(x, edge_index, edge_weight))
        x = F.silu(self.conv3(x, edge_index, edge_weight))
        x = self.conv4(x, edge_index, edge_weight)
        
        return torch.sigmoid(x)


# %%

if __name__ == '__main__':

    # An object oriented library for handling EPANET files in Python
    import epynet 
    
    # Import a custom tool for converting EPANET .inp files to networkx graphs
    from utils.epanet_loader import get_nx_graph
    
    # SCADA timeseries dataloader
    from utils.data_loader import dataGenerator
    
    # Set the path to the EPANET input file
    path_to_wdn = './data/ANYTOWN.inp'
    
    # Import the .inp file using the EPYNET library
    wdn = epynet.Network(path_to_wdn)
    
    # Solve hydraulic model for a single timestep
    wdn.solve()
    
    # Convert the file using a custom function, based on:
    # https://github.com/BME-SmartLab/GraphConvWat 
    G , pos , head = get_nx_graph(wdn, weight_mode='inv_pipe_length', get_head=True)
    
    # Load the training timeseries data 
    # This was generated by running the 'generate_dta.py' script for
    # the 'anytown.inp' of the GraphConvWat project of G. Hajgató et al.
    x_trn = np.load('./data/anytown-data/trn_x.npy')
    y_trn = np.load('./data/anytown-data/trn_y.npy')
    x_val = np.load('./data/anytown-data/vld_x.npy')
    y_val = np.load('./data/anytown-data/vld_y.npy')
    
    scale_y = np.std(y_trn)
    bias_y  = np.mean(y_trn)
    
    # ----------------
    # Hyper-parameters
    # ----------------
    batch_size    = 40
    learning_rate = 3e-4
    decay         = 6e-6
    shuffle       = False
    epochs        = 5
    
    # --------------
    # Training setup
    # --------------
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate training and test set data generators
    trn_gnrtr = dataGenerator(G, x_trn, y_trn, batch_size, shuffle)
    val_gnrtr = dataGenerator(G, x_val, y_val, batch_size, shuffle)
    
    # Instantiate a Chebysev Network GNN model
    model     = ChebNet(name           = 'ChebNet',
                        data_generator = trn_gnrtr,
                        device         = device, 
                        in_channels    = np.shape(x_trn)[-1], 
                        out_channels   = np.shape(y_trn)[-1],
                        data_scale     = scale_y, 
                        data_bias      = bias_y).to(device)
    
    # Instantiate an optimizer
    optimizer = torch.optim.Adam([dict(params=model.conv1.parameters(), weight_decay=decay),
                                  dict(params=model.conv2.parameters(), weight_decay=decay),
                                  dict(params=model.conv3.parameters(), weight_decay=decay),
                                  dict(params=model.conv4.parameters(), weight_decay=0)],
                                  lr  = learning_rate,
                                  eps = 1e-7)
    
    # Configure an early stopping callback
    estop    = EarlyStopping(min_delta=.00001, patience=30)
    
    # Initialise the best validation loss for saving the best model
    best_val_loss = np.inf
    
    # Train for the predefined number of epochs
    for epoch in range(1, epochs+1):
        
        # Start a stopwatch timer
        start_time = time.time()
        
        # Train a single epoch, passing the optimizer and current epoch number
        model.train_one_epoch(optimizer     = optimizer,
                              current_epoch = epoch)
        
        # Validate the model after the gradient update
        model.validate()
        
        # Update the model results for the current epoch
        model.update_results()
        
        # Print stats for the epoch and the execution time
        model.print_stats(time.time() - start_time)
        
        # If this is the best model
        if model.val_loss < best_val_loss:
            # We save it
            torch.save(model.state_dict(), './models/best_model.pt')
        
        # If model is not improving
        if estop.step(torch.tensor(model.val_loss)):
            print('Early stopping activated...')
            break
    
