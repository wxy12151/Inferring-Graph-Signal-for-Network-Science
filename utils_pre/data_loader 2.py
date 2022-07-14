#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

+----------------------------+
|                            |
|   D A T A   L O A D E R S  | 
|                            |
+----------------------------+

Created on Fri Jul 23 15:24:59 2021

@author: gardar
"""
import os 

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader
from torch_geometric.utils import from_networkx

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

# Random batch sampler that maintains sequential ordering for temporal learning tasks
from modules.torch_samplers import RandomBatchSampler

# Function to load the timeseries datasets of the BattLeDIM challenge
def battledimLoader(observed_nodes, n_nodes=782, path='./BattLeDIM/', file='2018_SCADA_Pressures.csv',
                    rescale=False, scale=None, bias=None,
                    mode='sensor_mask',task='reconstruction',n_timesteps=None):
    '''
    Function for loading the SCADA .csv datasets of the BattLeDIM competition
    and returning it in a dataformat suitable for the GNN model to ingest.

    Parameters
    ----------
    observed_nodes : list of ints
        A list of numerical values indicating the sensors nodal placement.
    n_nodes : int, optional
        Total no. of nodes in the network. The default is 782.
    path : str, optional
        Directory name containing SCADA data. The default is './BattLeDIM/'.
    file : str, optional
        Filename. The default is '2018_SCADA_Pressures.csv'.

    Returns
    -------
    result : np.array(n_obs,n_nodes,2)
        An array of size (n_observations x n_nodes x 2).
        These are then n number of 2-d matrices where the 1st dimension is
        nodal pressure value, and the 2nd dimension is a mask, 1 if the
        pressure value is present (at the observed nodes) and 0 if not
        
        E.g.:
        
        [21.57, 1    <- n1, pressure at node 1 is observed
         0.0  , 0    <- n2, pressure at node 2 is unknown
         0.0  , 0    <- n3
         22.43, 1    <- n4
         0.0  , 0    <- n5
         ...     ]   etc.
    '''
    # Read the file at the passed destination into a Pandas DataFrame
    df = pd.read_csv(str(path + file), sep=';', decimal=',')
    
    # Set the 'Timestamp' column as the index
    df = df.set_index('Timestamp')
    
    # Set the column names as the numeric list passed into the function
    # which states what nodes of the graphs are observed
    df.columns = observed_nodes
    
    # User has option of rescaling the imported data
    if rescale:
        df = (df - bias) / scale
        
    # Generate a temporary image of the DataFrame, that's been filled with zeros
    # at the un-observed nodes
    temp = df.T.reindex(list(range(1,n_nodes+1)),fill_value=0.0)
        
    if mode=='sensor_mask':

        # Create a "mask" array, that's set to 1 at the observed nodes and 0 otherwise
        arr2 = np.array(temp.mask(temp>0.0,1).astype('int'))

        # Create a numpy array from the temporary image
        arr1 = np.array(temp)

        # Stack and transpose the observation and mask arrays
        result = np.stack((arr1,arr2),axis=0).T
        
    # Returns a (n_observations, n_nodes, n_timesteps) feature vector (x) where the 3rd dimension
    # is the timesteps t, t-1, t-2 ... t-n leading to the observation to be predicted, at t+1
    if mode=='n_timesteps':
        
        x_df = temp.T                                                   # The feature dataframe (missing observations)

        if task == 'prediction':                                        # If we're doing prediction we set the
            n_samples = len(x_df)                                       # no.of samples as length of DF
            
        elif task == 'reconstruction':                                  # If we're doing reconstruction we set the
            n_samples = len(x_df)+1                                     # no.of samples as length of DF + 1 due to
                                                                        # slicing
                
        window_start = 0                                                # Set the start/end of the rolling window
        window_end   = n_timesteps                                      # to be used to retrieve t-n timesteps for x

        x_ = []                                                         # Initialise temp x_ and y_ lists
                                                                        # to contain our features and vectors

        for i in range(n_timesteps,n_samples):                          # For each training sample
            x_arr = (x_df.iloc[window_start:window_end].to_numpy().T)   # Add the t-n partial pressure signals
            x_.append(np.flip(x_arr,axis=1))                            # Flip the order so that t is at index 0
            window_start+=1                                             # Increment the
            window_end  +=1                                             # rolling window

        result = np.array(x_)                                           # Dump our lists to an np.array
        
    
    # Return the results
    return result

# Function to clean the nominal pressure dataframe
def dataCleaner(pressure_df, observed_nodes,
                rescale=None, mode='sensor_mask', task='reconstruction', n_timesteps=None):
    '''
    Function for cleaning the pressure dataframes obtained by simulation of the
    nominal system model supplied with the BattLeDIM competition.
    The output format is suitable for ingestion by the GNN model.
    
    Parameters
    ----------
    pressure_df : pd.DataFrame
        Pandas dataframe where:
            columns (x) = nodes
            index   (y) = observations
    sensor_list : list of ints
        A list of numerical values indicating the sensors nodal placement..
    scaling : str
        'standard' - standard scaling
        'minmax'   - min/max scaling
    mode : str
        'sensor_mask' - A per timestep stacked feature output np.array as per below
        'n_timesteps' - A t-n timestep stacked feature output np.array as per below
    task : str
        'reconstruction' - Returns y[t]   for x[t],x[t-1]...x[t-n] timesteps
        'prediction'     - Returns y[t+1] for x[t],x[t-1]...x[t-n] timesteps
        
    Returns
    -------
    if mode='sensor_mask'
    
    x : np.array(n_obs,n_nodes,2)
        The incomplete pressure signal matrix w/ 'n' number of observations.
        This is the feature vector (x) for the GNN model
        
        x =
        [21.57, 1    <- n1, pressure at node 1 is observed
         0.0  , 0    <- n2, pressure at node 2 is unknown
         0.0  , 0    <- n3, ... unknown
         22.43, 1    <- n4, ... observed
         0.0  , 0    <- n5, ... unknown
         ...     ]   etc.
         
    if mode='n_timesteps'
    
    x : np.array(n_obs,n_nodes,n_timesteps)
        The incomplete pressure signal matrix w/ 'n' number of observations, for n timesteps
        This is the feature vector (x) for the GNN model
        
        x =
        [21.57, 22.81, 23.13, ... , t-n    <- n1, pressure at node 1 is observed
         0.0  , 0.0  , 0.0  , ... , t-n    <- n2, pressure at node 2 is unknown
         0.0  , 0.0  , 0.0  , ... , t-n    <- n3, ... unknown
         22.43, 22.51, 23.41, ... , t-n    <- n4, ... observed
         0.0  , 0.0  , 0.0  , ... , t-n    <- n5, ... unknown
         ...     ]   etc.
        
    y : np.array(n_obs,n_nodes,2)
        The complete pressure signal matrix w/ 'n' number of observations.
        With this we may train the GNN in a supervised manner.
        
        y =
        [21.57    <- n1, all values are observed
         21.89    <- n2,
         22.17    <- n3
         22.43    <- n4
         23.79    <- n5
         ...  ]   etc.
        
    '''
    # The number of nodes in the passed dataframe
    n_nodes = len(pressure_df.columns)
    
    # Rename the columns (n1, n2, ...) to numerical values (1, 2, ...)
    pressure_df.columns = [number for number in range(1,n_nodes+1)]
    
    # Perform scaling on the initial Pandas Dataframe for brevity
    # This is less trivial than applying it on the later generated numpy arrays
    
    # Standard scale:
    if rescale == 'standard':
        _avg        = pressure_df.stack().mean()        # Calc. avg. over entire df.
        _std        = pressure_df.stack().std(ddof=0)   # Calc. std.. over entire df.
        bias        = _avg                              # Avg. is the scaling bias
        scale       = _std                              # Std.dev. is the scaling range
        pressure_df = (pressure_df - bias) / scale      # Scale to range
        
    # Min/max scaling (normalising):
    elif rescale == 'minmax':
        _min        = min(pressure_df.min())            # Find the absolute minimum value
        _max        = max(pressure_df.max())            # Find the absolute maximum value
        _rng        = _max - _min                       # Calculate the difference between (range)
        bias        = _min                              # Scaling bias is the min value
        scale       = _rng                              # Scaling range is the min-max range
        pressure_df = (pressure_df - bias) / scale      # Scale to range
        
    # Perform no scaling
    else:
        bias        = None
        scale       = None
    
    # DataFrame where the index is the node number holding the sensor and the value is set to 1
    sensor_df = pd.DataFrame(data=[1 for i in observed_nodes],index=observed_nodes)
    
    # Filled single row of DataFrame with the complete number of nodes, the unmonitored nodes are set to 0
    sensor_df = sensor_df.reindex(list(range(1,n_nodes+1)),fill_value=0)
    
    # Find the number of rows in the DataFrame to be masked...
    n_rows = len(pressure_df)
    
    # ... and complete a mask DataFrame, where all the observations to keep are set to 1 and the rest to 0
    mask_df = sensor_df.T.append([sensor_df.T for i in range(n_rows-1)],ignore_index=True)
    
    # Enforce matching indices of the two DataFrames to be broadcast together
    mask_df.index = pressure_df.index
    
    # Returns a (n_observations, n_nodes, 2) feature vector (x) where the 3rd dimension is a 0/1 mask
    # of the observed nodes
    if mode=='sensor_mask':
        
        # Generating the incomplete feature matrix (x)
        x_mask = np.array(mask_df)
        x_arr  = np.array(pressure_df.where(cond=mask_df==1,other = 0.0))
        x      = np.stack((x_arr,x_mask),axis=2)

        # Generating the complete label matrix (y)
        y_arr  = np.array(pressure_df)
        y      = np.stack((y_arr, ),axis=2)
    
    # Returns a (n_observations, n_nodes, n_timesteps) feature vector (x) where the 3rd dimension
    # is the timesteps t, t-1, t-2 ... t-n leading to the observation to be predicted, at t+1
    if mode=='n_timesteps':
        
        x_df         = pressure_df.where(cond=mask_df==1,other = 0.0)   # The feature dataframe (missing observations)
        y_df         = pressure_df                                      # The label dataframe (complete observation)
        
        if task == 'prediction':                                        # If we're doing prediction we set the
            n_samples = len(x_df)                                       # no.of samples as length of DF
            
        elif task == 'reconstruction':                                  # If we're doing reconstruction we set the
            n_samples = len(x_df)+1                                     # no.of samples as length of DF + 1 due to
                                                                        # slicing
                
        window_start = 0                                                # Set the start/end of the rolling window
        window_end   = n_timesteps                                      # to be used to retrieve t-n timesteps for x

        x_ = []                                                         # Initialise temp x_ and y_ lists
        y_ = []                                                         # to contain our features and vectors

        for i in range(n_timesteps,n_samples):                          # For each training sample
            x_arr = (x_df.iloc[window_start:window_end].to_numpy().T)   # Add the t-n partial pressure signals
            x_.append(np.flip(x_arr,axis=1))                            # Flip the order so that t is at index 0
                                                                        # t-1 is at index 1, and so on
            
            if task == 'prediction':                                    # For prediction
                y_.append(y_df.iloc[i])                                 # Add complete observation at t+1 as label
                
            elif task == 'reconstruction':                              # For reconstruction
                y_.append(y_df.iloc[i-1])                               # Add complete observation at t as label
            
            window_start+=1                                             # Increment the
            window_end  +=1                                             # rolling window

        x = np.array(x_)                                                # Dump our lists
        y = np.array(y_)                                                # to arrays
        
        row,col = y.shape                                               # Reshape the label array y
        shape   = (row,col,1)                                           # so its dimensions are (n_observations, 1)
        y = y.reshape(shape)                                            # not (n_observations, )
        
    return x,y,scale,bias                                               # Return the features, labels, scale & bias
    


# Function that embeds the x-y labels on the graph and returns a DataLoader obj.
def dataGenerator(G, features, labels, batch_size, drop_last):
    '''
    Function that embeds x and y labels on a graph and returns a torch
    data loader object

    Parameters
    ----------
    G : networkx graph
        The graph.
    features : numpy array
        The x-features.
    labels : numpy array
        The y-labels.
    batch_size : int
        The batch size.
    drop_last : bool
        Leave out the last fold if inconsistent with batch_size?

    Returns
    -------
    generator : TYPE
        DataLoader object (generator for training).

    '''
    # Initialise empty data list
    data    = []
    
    # For each array in the passed sets
    for x, y in zip(features, labels):
        graph   = from_networkx(G)  # Turn graph into tensor
        graph.x = torch.Tensor(x)   # Embed features on graph
        graph.y = torch.Tensor(y)   # Embed labels to the graph
        data.append(graph)          # Append the tensorised graph reps to the data list
    
    # Generate a sampler for picking sequential mini-batches at random from dataset
    sampler   = RandomBatchSampler(data, batch_size, drop_last=False)
    
    # Construct a generator (data loader) from the list of graphs
    generator = DataLoader(data, 
                           batch_sampler = sampler)
                           #batch_size = batch_size, 
                           #shuffle    = shuffle)
    
    return generator

def embedSignalOnGraph(G, signal):
    '''
    Function that embeds a signal, e.g. a partial one, on a given graph G

    Parameters
    ----------
    G : networkx graph
        The graph.
    signal : numpy array
        The x-features to embed on the graph.

    Returns
    -------
    graph : TYPE
        DESCRIPTION.

    '''
    # Generate a torch tensor object from the networkx graph
    graph  = from_networkx(G)
    # Embed the signal to the x-features of the tensor
    graph.x= torch.Tensor(signal)
    
    # Return the graph tensor object
    return graph

def rescaleSignal(signal, scale, bias):
    '''
    Graph signals have often times been normalised or standardised
    Given the scale and bias (e.g. mean and std in standard scaling, or min.
    val and range during normalisation) the signal has been scaled as per:
        (signal - bias) / scale
    So, given a signal, bias and scale we may recalculate the initial signal 
    as per:
        (signal * scale) + scale

    Parameters
    ----------
    signal : numpy array
        The signal to rescale.
    scale : scalar
        The signal scale.
    bias : scalar
        The signal bias.

    Returns
    -------
    (signal * scale) + bias.

    '''    
    return((np.dot(signal,scale))+bias)

def predictionTaskDataSplitter(x, y, n_timesteps):
    '''
    Splitting a given dataset (x,y) where 'x' are features at n-number of timesteps and 'y' label or target state for the given 'x'
    The parameter n_timesteps decides how many 'x' are behind a given 'y'
    For prediction, if we are to predict y_hat( t+1 ), with n=3, then the splitter returns x( t, t-1, t-2 ), y(t+1)

    Parameters
    ----------
    x : numpy array
        The timeseries signal array
    y : numpy array
        The target timeseries signal array
    n_timesteps : int
        Number of timesteps in x that correspont to target state in y

    Returns
    -------
    x, y where x(t, t-1, ... t-n) for each y(t+1)

    '''
    window_start = 0
    window_end = n_timesteps
    n_samples = len(y)
    x_new = []
    y_new = []
    
    for i in range(n_timesteps, n_samples):
        x_new.append( np.array( x[window_start:window_end] ) )
        y_new.append( y[i] )
        window_start += 1
        window_end   += 1
        
    return np.array(x_new) , np.array(y_new)
