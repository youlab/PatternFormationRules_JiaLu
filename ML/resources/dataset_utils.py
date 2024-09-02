import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# Dataset of param and reconstructed 1D profile
class param_recon_dataset(Dataset):
    def __init__(self, param_array, data_array):
        self.param_array = param_array
        self.data_array = data_array

    def __len__(self):
        return len(self.param_array)

    def __getitem__(self, idx):
        return self.param_array[idx], self.data_array[idx]
    
# Dataset of param and latent variable
class param_latent_dataset(Dataset):
    
    def __init__(self, param_array, latent_array):
        self.param_array = param_array
        self.latent_array = latent_array

    def __len__(self):
        return len(self.param_array)

    def __getitem__(self, idx):

        return self.param_array[idx], self.latent_array[idx]

