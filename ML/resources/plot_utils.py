import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from joblib import Parallel, delayed
from sklearn.metrics import r2_score

# Plot R2 for orignal and prediction, and save
def plot_R2(ori, pred, filename):
    # Inputs
    # ori size: N_1 * seq_length - 2D, on CPU
    # pred size: N_2 * seq_length - 2D, on CPU
    # filename: full name including folder path

    # Flatten data - take 500 data points
    flattened_ori = ori[0: 500].flatten()
    flattened_pred = pred[0: 500].flatten()

    # Plot
    fig, axs = plt.subplots(1, 1, figsize=(2, 2))
    # scattered
    axs.scatter(flattened_ori, flattened_pred, s=0.1, color='orange', alpha=0.5)
    # line
    axs.plot([flattened_ori.min(), flattened_ori.max()], [flattened_ori.min(), flattened_ori.max()], 'blue',alpha=0.5)
    # plot setup
    axs.set_xlim(0, 1)
    axs.set_ylim(0, 1)
    axs.set_aspect('equal', adjustable='box')
    axs.set_xlabel('Original')
    axs.set_ylabel('Reconstructed')
    axs.set_title('R2')
    
    # Print R2 in figure
    r2 = r2_score(flattened_ori, flattened_pred)
    axs.text(0.05, 0.95, f'R^2 = {r2:.4f}', transform=axs.transAxes, verticalalignment='top')

    plt.tight_layout()
    # Save
    plt.savefig(filename, transparent=True)
    plt.show()
    

# Plot a single example
def plot_profiles(data):
    # Input: a 1D profile of all 3 channels, size 3 * seq_length
    # Not needed to normalize

    seq_length = int(data.shape[0]/3)
    data_C = data[0:seq_length]
    data_RFP = data[seq_length:seq_length*2]
    data_CFP = data[seq_length*2:seq_length*3]
    
    # Remove NaN and negative values
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

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))
    axs[0].plot(data_C, label='C')
    axs[1].plot(data_RFP, label='RFP')
    axs[2].plot(data_CFP, label='CFP')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.tight_layout()
    plt.show()
    
def plot_save_profiles(data, output_path, filename):
    
    # Input: a 1D profile of all 3 channels, size 3 * seq_length
    # Not needed to normalize
    # output_path: folder to save figure
    # filename: does not include full path

    seq_length = int(data.shape[0]/3)
    data_C = data[0:seq_length]
    data_RFP = data[seq_length:seq_length*2]
    data_CFP = data[seq_length*2:seq_length*3]
    
    # Remove NaN and negative values
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

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))
    axs[0].plot(data_C, label='C')
    axs[1].plot(data_RFP, label='RFP')
    axs[2].plot(data_CFP, label='CFP')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    
    
    # Save and show plot
    plt.tight_layout()
    filename = os.path.join(output_path, filename)
    plt.savefig(filename)
    plt.show()
    
# Plot RFP channel
def plot_save_RFP(data, dir_path, filename):
    # data: all 3 channels of a data point, size 3 * seq_length
    # output_path: folder to save figure
    # filename: does not include full path
    
    seq_length = int(data.shape[0]/3)
    data_RFP = data[seq_length:seq_length*2]

    # Remove NaN and negative values
    data_RFP[data_RFP < 0] = 0
    data_RFP[np.isnan(data_RFP)] = 0
    
    # Construct the full RFP profile
    flipped_data_RFP = np.flip(data_RFP)
    data_RFP = np.concatenate((flipped_data_RFP, data_RFP))

    # Normalize
    data_RFP = (data_RFP - np.min(data_RFP)) / (np.max(data_RFP) - np.min(data_RFP))

    # Plot
    fig, axs = plt.subplots(1, 1, figsize=(3, 2))
    axs.plot(data_RFP, label='RFP')
    plt.tight_layout()
    
    # Save and show plot
    filename = os.path.join(output_path, filename)
    plt.savefig(filename, transparent=True)
    plt.show()
    


# Plot 3 channel profiles, with peaks
def plot_profiles_peak(data):
    # data: size 3 * seq_length
    
    seq_length = int(data.shape[0]/3)
    data_C = data[0:seq_length]
    data_RFP = data[seq_length:seq_length*2]
    data_CFP = data[seq_length*2:seq_length*3]
    
    # Remove NaN and negative values
    data_C[data_C < 0] = 0
    data_RFP[data_RFP < 0] = 0
    data_CFP[data_CFP < 0] = 0

    data_C[np.isnan(data_C)] = 0
    data_RFP[np.isnan(data_RFP)] = 0
    data_CFP[np.isnan(data_CFP)] = 0
    
    # Construct the full profiles
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
    
     # Get pattern type
    C_peaks, _ = signal.find_peaks(data_C, distance=5, width = 3, height = 0.1*np.max(data_C), prominence=0.03*np.max(data_C))
    RFP_peaks, _ = signal.find_peaks(data_RFP, distance=5, width = 3, height = 0.03*np.max(data_RFP), prominence=0.03*np.max(data_RFP))
    CFP_peaks, _ = signal.find_peaks(data_CFP, distance=5, width = 3, height = 0.03*np.max(data_CFP), prominence=0.03*np.max(data_CFP))
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))
    axs[0].plot(data_C, label='C')
    axs[0].plot(C_peaks, data_C[C_peaks], 'x')
    axs[1].plot(data_RFP, label='RFP')
    axs[1].plot(RFP_peaks, data_RFP[RFP_peaks], 'x')
    axs[2].plot(data_CFP, label='CFP')
    axs[2].plot(CFP_peaks, data_CFP[CFP_peaks], 'x')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.tight_layout()
    plt.show()
    
# Plot RFP profile with peaks
def plot_profiles_peak_1channel(data):
    # data: seq_legth
    
    # Remove NaN and negative values
    data[data < 0] = 0
    data[np.isnan(data)] = 0
    
    # Construct the full RFP profile
    flipped_data = np.flip(data)
    data = np.concatenate((flipped_data, data))

    # Normalize
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Get pattern type of the originals
    peaks, _ = signal.find_peaks(data, distance=5, width = 3, height = 0.03*np.max(data), prominence=0.03*np.max(data))
    
    # Plot
    fig, axs = plt.subplots(1, 1, figsize=(4, 2))
    axs.plot(data)
    axs.plot(peaks, data[peaks], 'x')
    plt.tight_layout()
    plt.show()


# Plot RFP profile with peaks and save
def plot_save_profiles_peak_1channel(data, output_path, filename):
    # data: seq_legth
    # output_path: folder to save figure
    # filename: does not contain the folder path

    # Remove NaN and negative values
    data[data < 0] = 0
    data[np.isnan(data)] = 0
    
    # Construct the full RFP profile
    flipped_data = np.flip(data)
    data = np.concatenate((flipped_data, data))

    # Normalize
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Get pattern type of the originals
    peaks, _ = signal.find_peaks(data, distance=5, width = 3, height = 0.03*np.max(data), prominence=0.03*np.max(data))
    
    # Plot
    fig, axs = plt.subplots(1, 1, figsize=(4, 2))
    X = np.linspace(-0.5*len(data), 0.5*len(data), len(data))
    axs.plot(X, data)
    axs.plot(peaks - 0.5*len(data), data[peaks], 'x')
    plt.tight_layout()
    
    # Save and show plot
    filename = os.path.join(output_path, filename)
    plt.savefig(filename, transparent=True)
    plt.show()
        

# Compare ML prediction and PDE validation
def plot_save_profiles_peak_1channel_compare(data_1, data_2, dir_path, filename):
    # data_1 is the ML predicted profile
    # data_2 is the PDE validation profile
    
    # Remove NaN and negative values
    data_1[data_1 < 0] = 0
    data_2[data_2 < 0] = 0

    data_1[np.isnan(data_1)] = 0
    data_2[np.isnan(data_2)] = 0
    
    # Construct full RFP profiles
    flipped_data_1 = np.flip(data_1)
    flipped_data_2 = np.flip(data_2)

    data_1 = np.concatenate((flipped_data_1, data_1))
    data_2 = np.concatenate((flipped_data_2, data_2))
        
    
    # Normalize
    data_1 = (data_1 - np.min(data_1)) / (np.max(data_1) - np.min(data_1))
    data_2 = (data_2 - np.min(data_2)) / (np.max(data_2) - np.min(data_2))

    
    # Get pattern type of the originals
    peaks_1, _ = signal.find_peaks(data_1, distance=5, width = 3, height = 0.03*np.max(data_1), prominence=0.03*np.max(data_1))
    peaks_2, _ = signal.find_peaks(data_2, distance=5, width = 3, height = 0.03*np.max(data_2), prominence=0.03*np.max(data_2))

    fig, axs = plt.subplots(1, 1, figsize=(4, 2))
    X = np.linspace(-0.5*len(data_1), 0.5*len(data_1), len(data_1))
    axs.plot(X, data_1, 'blue', label='ML prediction')
    axs.plot(X, data_2, 'orange', label = 'PDE simulation')

    plt.tight_layout()
    # Save and show plot
    filename = os.path.join(dir_path, filename)
    plt.savefig(filename,transparent=True)
    plt.show()
    

def plot_pattern_type_hist(C_type_list, RFP_type_list, CFP_type_list):
    # inputs are lists
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plotting each list in a subplot
    axs[0].hist(C_type_list, bins=len(set(C_type_list)), edgecolor='black')
    axs[0].set_title('C type histogram')
    axs[0].set_xlabel('Type')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(RFP_type_list, bins=len(set(RFP_type_list)), edgecolor='black')
    axs[1].set_title('RFP type histogram')
    axs[1].set_xlabel('Type')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(CFP_type_list, bins=len(set(CFP_type_list)), edgecolor='black')
    axs[2].set_title('CFP type histogram')
    axs[2].set_xlabel('Type')
    axs[2].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()
    plt.show()
    


