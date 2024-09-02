import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import itertools
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from scipy.io import loadmat
import scipy.signal as signal

def peak_to_ring_num(peak_num_list):
    # Input: number of peaks, numpy array
    # Output: number of rings
    
    ring_num_list = np.ceil(peak_num_list/2)
    
    return ring_num_list

def get_pattern_type(data_3channel):
    # input size: N * (3 * sequence length) - 2D
    
    C_type_list = []
    RFP_type_list = []
    CFP_type_list = []
    
    C_peaks_list = []
    RFP_peaks_list = []
    CFP_peaks_list = []
    
    seq_length = int(data_3channel.shape[1]/3)
    
    for i in tqdm(range(0, len(data_3channel))):
        
        data = data_3channel[i]
        data_C = data[0:seq_length]
        data_RFP = data[seq_length:2*seq_length]
        data_CFP = data[2*seq_length:3*seq_length]
        
        # Filter off negative values and Nan
        data_C[data_C < 0] = 0
        data_RFP[data_RFP < 0] = 0
        data_CFP[data_CFP < 0] = 0

        data_C[np.isnan(data_C)] = 0
        data_RFP[np.isnan(data_RFP)] = 0
        data_CFP[np.isnan(data_CFP)] = 0
        
        # Construct the full 1D profile
        flipped_data_C = np.flip(data_C)
        flipped_data_RFP = np.flip(data_RFP)
        flipped_data_CFP = np.flip(data_CFP)

        data_C = np.concatenate((flipped_data_C, data_C))
        data_RFP = np.concatenate((flipped_data_RFP, data_RFP))
        data_CFP = np.concatenate((flipped_data_CFP, data_CFP))
        
        # Normalize
        data_C = (data_C - np.min(data_C)) / (np.max(data_C) - np.min(data_C))
        data_RFP = (data_RFP - np.min(data_RFP)) / (np.max(data_RFP) - np.min(data_RFP))
        data_CFP = (data_CFP - np.min(data_CFP)) / (np.max(data_CFP) - np.min(data_CFP))
    
        # Get pattern types
        C_peaks, _ = signal.find_peaks(data_C, distance=5, width = 3, height = 0.1*np.max(data_C), prominence=0.03*np.max(data_C))
        RFP_peaks, _ = signal.find_peaks(data_RFP, distance=5, width = 3, height = 0.03*np.max(data_RFP), prominence=0.03*np.max(data_RFP))
        CFP_peaks, _ = signal.find_peaks(data_CFP, distance=5, width = 3, height = 0.03*np.max(data_CFP), prominence=0.03*np.max(data_CFP))
 
        C_type = len(C_peaks)
        RFP_type = len(RFP_peaks)
        CFP_type = len(CFP_peaks)
        
        C_peaks_list.append(C_peaks)
        RFP_peaks_list.append(RFP_peaks)
        CFP_peaks_list.append(CFP_peaks)
        
        C_type_list.append(C_type)
        RFP_type_list.append(RFP_type)
        CFP_type_list.append(CFP_type)
        
    return C_peaks_list, RFP_peaks_list, CFP_peaks_list, C_type_list, RFP_type_list, CFP_type_list


def get_RFP_type(data_RFP):
    # Input is RFP profile
    # Input size: N * sequence_length
    
    RFP_type_list = []
    RFP_peaks_list = []
    
    for i in tqdm(range(0, len(data_RFP))):
        
        data = data_RFP[i, 0, :]
        
        # Remove NaN and negative values
        data[data < 0] = 0
        data[np.isnan(data)] = 0
        
        # Construct full RFP profile
        flipped_data = np.flip(data)
        data = np.concatenate((flipped_data, data))

        # Normalize
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        if not np.any(data): # If data is all 0s
            RFP_type = 0
            RFP_peaks = []

        else:
            # Get pattern type
            RFP_peaks, _ = signal.find_peaks(data, distance=5, width = 3, height = 0.03*np.max(data), prominence=0.03*np.max(data))
            RFP_type = len(RFP_peaks)

        RFP_peaks_list.append(RFP_peaks)
        RFP_type_list.append(RFP_type)
            
    return RFP_peaks_list, RFP_type_list


def get_RFP_type_single(data):
    # Input is RFP profile
    # Input size: sequence_length - 1D
    
    
    # Remove NaN and negative values
    data[data < 0] = 0
    data[np.isnan(data)] = 0
    
    # Construct full 1D profile
    flipped_data = np.flip(data)
    data = np.concatenate((flipped_data, data))
    
    # Normalize
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    if not np.any(data): # if data is all 0s
        RFP_type = 0
        RFP_peaks = []

    else:
        # Get pattern type of the originals
        RFP_peaks, _ = signal.find_peaks(data, distance = 5, width = 3, height = 0.03*np.max(data), prominence = 0.03*np.max(data))
        RFP_type = len(RFP_peaks)

    return RFP_peaks, RFP_type


# Use L1 norm to measure smoothness
def check_smoothness(data):
    # Input is RFP profile
    # Input size: sequence_length

    # Calculate the derivative
    x = np.linspace(0, len(data)+1, len(data))
    total_variation = np.sum(np.abs(np.diff(data)))
    
    # Determine smoothness
    threshold = 0.1
    if total_variation < threshold:
        if_smooth = 0
    else:
        if_smooth = 1
    return if_smooth, total_variation
    
# Use L1 norm to measure smoothness
def check_smoothness_dataset(dataset):
    if_smooth_list = []
    
    for i in range(0, len(dataset)):
        data = dataset[i, 0, :]
        if_smooth, _ = check_smoothness(data)
        if (data[-1] > 0.05 * np.max(data)):
            if_smooth = 0
        if_smooth_list.append(if_smooth)
        
    return if_smooth_list
    
# Use trained VAE encoder to get latent variables, return z
def get_latent(model, data):

    model.eval()
    
    z_values = []  
    
    with torch.no_grad():
        
        for d in data:
            
            mu, log_var = model.encoder(d.cuda().unsqueeze(0))  # Add batch dimension
            z = model.reparameterize(mu, log_var)
            z = z.detach().cpu()
            z_values.append(z.unsqueeze(1))  
                
    return torch.cat(z_values, dim=0)

# Normalize parameters to [0, 1]
def scale_feature(value, min_val, max_val, option):
    # Input size: a single number
    
    if option == "linear":
        out = (value - min_val) / (max_val - min_val)
        out = np.round(out, 18)
        return out
    
    elif option == "exp":
        out = (np.log(value) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
        out = np.round(out, 18)
        return out
    
    elif option == "int":
        out = (value - min_val) / (max_val - min_val)
        out = np.round(out, 18)
        return out
    
    else:
        raise ValueError(f"Unknown scaling option: {option}")

        
# Normalize parameter dataset to [0, 1]
def scale_dataset(dataset, ranges, opt):
    # Dataset size: N * num_params
    # ranges: a dictionary - {param_num: [min, max]}
    # opt: list
    
    scaled_dataset = np.zeros_like(dataset)
    
    # Process by column
    for i, (key, range_vals) in enumerate(ranges.items()):
    
        min_val, max_val = range_vals
        scaling_option = opt[i]
        
        scaled_dataset[:, i] = [scale_feature(value, min_val, max_val, scaling_option) for value in dataset[:, i]]
        
    return scaled_dataset


# Convert normalized parameters to actual scales
def scaleback_feature(value, min_val, max_val, option):
    # Input size: a single number

    if option == "linear":
        out = value * (max_val - min_val) + min_val
        out = np.round(out, 18)
    
    elif option == "exp":
        out =  np.exp(np.log(min_val) + value * (np.log(max_val) - np.log(min_val)))
        out = np.round(out, 18)
        
    return out

# Convert normalized parameters to actual scales
def scaleback_dataset(dataset, ranges, opt):
    # Dataset size: N * num_params
    # ranges: a dictionary - {param_num: [min, max]}
    # opt: list
    
    scaled_dataset = np.zeros_like(dataset)
    
    # Process by column
    for i, (key, range_vals) in enumerate(ranges.items()):
        
        min_val, max_val = range_vals
        scaling_option = opt[i]
        scaled_dataset[:, i] = [scaleback_feature(value, min_val, max_val, scaling_option) for value in dataset[:, i]]
        
    return scaled_dataset
    
    
# Sample on orignal scale
def random_sampling(ranges, opt, col_names, n_samples):
    # ranges: dictionary
    # opt: list
    # n_samples: number of parameter combinations to generate
    
    rand_params = []
    
    for i in range(0, len(ranges)):
        param = list(col_names)[i]
        rand_range = ranges[param]
        rand_opt = opt[i]
        
        if rand_opt == 'exp':
            lower_bound = max(rand_range[0], 1e-8)
            upper_bound = rand_range[1]
            
            exponents = np.random.uniform(low=np.log(lower_bound), high=np.log(upper_bound), size=n_samples)
            rand_col = np.exp(exponents)
            
        elif rand_opt == 'linear':
            lower_bound = max(rand_range[0], 1e-8)
            
            rand_col = np.random.uniform(low=lower_bound, high=rand_range[1], size=n_samples)
            
        elif rand_opt == 'int':
            rand_col =  np.random.choice([1,2,3,4,5,6,7,8,9,10], size=n_samples)

        if rand_opt in ['exp', 'linear']: # round to .8 to be consistant with txt fmt
            rand_col = np.round(rand_col, 8)
            
        rand_params.append(rand_col)
        
    out = np.concatenate(rand_params, axis=0)
    out = out.reshape([len(col_names), n_samples]).T
    
    return out

# Sample normalized params [0, 1]
def random_norm_sampling(n_samples, num_params):

    lower_bound = 1e-8
    upper_bound = 1
    
    out = np.random.uniform(low=lower_bound, high=upper_bound, size=[n_samples, num_params])
    out = np.round(out, 8)
    
    return out
