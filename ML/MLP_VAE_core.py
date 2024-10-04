import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.init as init


class CustomDataset(Dataset):
    def __init__(self, params, outputs, types):
        self.params = params
        self.outputs = outputs
        self.types = types
        
    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx], self.outputs[idx], self.types[idx]
    

class CustomDataset_with_type(Dataset):
    def __init__(self, input_data, output_data, output_types):
        self.input_data = input_data
        self.output_data = output_data
        self.output_types = output_types
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx], self.output_types[idx]
    
class MLP(nn.Module):
    def __init__(self, input_dim, latent_dim, seed):
        super(MLP, self).__init__()
        self.input_dim = input_dim # number of parameters
        self.latent_dim = latent_dim
        self.seed = seed
        
        if latent_dim == 2:
            self.fc1 = nn.Linear(input_dim, 16) 
            self.fc2 = nn.Linear(16, 64)       
            self.fc3 = nn.Linear(64, 128)       
            self.fc4 = nn.Linear(128, 64)        
            self.fc5 = nn.Linear(64, 32)     
            self.fc6 = nn.Linear(32, 16)      
            self.fc_mean = nn.Linear(16, latent_dim) 
            self.fc_logvar = nn.Linear(16, latent_dim) 
            
        elif latent_dim == 4:
            self.fc1 = nn.Linear(input_dim, 16) 
            self.fc2 = nn.Linear(16, 64)       
            self.fc3 = nn.Linear(64, 128)       
            self.fc4 = nn.Linear(128, 128)        
            self.fc5 = nn.Linear(128, 64)     
            self.fc6 = nn.Linear(64, 32)      
            self.fc_mean = nn.Linear(32, latent_dim) 
            self.fc_logvar = nn.Linear(32, latent_dim) 
            
        elif latent_dim == 8:
            self.fc1 = nn.Linear(input_dim, 32) 
            self.fc2 = nn.Linear(32, 64)       
            self.fc3 = nn.Linear(64, 128)        
            self.fc4 = nn.Linear(128, 128)       
            self.fc5 = nn.Linear(128, 64)       
            self.fc6 = nn.Linear(64, 32)       
            self.fc_mean = nn.Linear(32, latent_dim) 
            self.fc_logvar = nn.Linear(32, latent_dim) 

        elif latent_dim == 16:
            self.fc1 = nn.Linear(input_dim, 32) 
            self.fc2 = nn.Linear(32, 64)        
            self.fc3 = nn.Linear(64, 128)        
            self.fc4 = nn.Linear(128, 128)       
            self.fc5 = nn.Linear(128, 64)       
            self.fc6 = nn.Linear(64, 32)       
            self.fc_mean = nn.Linear(32, latent_dim) 
            self.fc_logvar = nn.Linear(32, latent_dim) 
            
        elif latent_dim == 32:
            self.fc1 = nn.Linear(input_dim, 32) 
            self.fc2 = nn.Linear(32, 64)        
            self.fc3 = nn.Linear(64, 128)        
            self.fc4 = nn.Linear(128, 128)       
            self.fc5 = nn.Linear(128, 128)       
            self.fc6 = nn.Linear(128, 64)       
            self.fc_mean = nn.Linear(64, latent_dim) 
            self.fc_logvar = nn.Linear(64, latent_dim) 
            
        elif latent_dim == 64:
            self.fc1 = nn.Linear(input_dim, 32) 
            self.fc2 = nn.Linear(32, 64)        
            self.fc3 = nn.Linear(64, 128)        
            self.fc4 = nn.Linear(128, 128)       
            self.fc5 = nn.Linear(128, 128)       
            self.fc6 = nn.Linear(128, 64)       
            self.fc_mean = nn.Linear(64, latent_dim) 
            self.fc_logvar = nn.Linear(64, latent_dim) 
       
        self.relu = nn.ReLU()                
        self._initialize_weights()

    def _initialize_weights(self):
        torch.manual_seed(self.seed)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar



# Define the combined model consisting of the MLP and the decoder of the VAE
class CombinedModel(nn.Module):
    def __init__(self, input_dim, latent_dim, decoder, seed):
        super(CombinedModel, self).__init__()
        self.mlp = MLP(input_dim, latent_dim, seed)
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.mlp(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        
        return reconstruction, mean, logvar

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_combined(model, dataloader, optimizer, criterion, alpha, device):
    model.train()
    running_loss = 0
    for params, data, types in dataloader:

        params = params.to(device)
        data = data.to(device)
    
        optimizer.zero_grad()
        
        reconstruction, mean, logvar = model(params)
        
        # Compute loss - MSE + alpha * KL-divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        recon_loss = criterion(reconstruction, data)
        loss = recon_loss + alpha * kl_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(params)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_combined(model, dataloader, criterion, alpha, device):
    
    model.eval()
    running_loss = 0
    
    with torch.no_grad():
        for params, data, types in dataloader:

            params = params.to(device)
            data = data.to(device)
    
            reconstruction, mean, logvar = model(params)
        
            # Compute loss 
            # KL loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            # Recon loss
            recon_loss = criterion(reconstruction, data)
            loss = recon_loss + alpha * kl_loss
            
            # Record loss
            running_loss += loss.item() * len(params)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def test_combined(model, dataloader, criterion, alpha, device):
    
    model.eval()
    running_loss = 0
    
    with torch.no_grad():
        for params, data, types in dataloader:

            params = params.to(device)
            data = data.to(device)
    
            reconstruction, mean, logvar = model(params)
            
            # # Smooth ML recon
            # smoothed_reconstruction = torch.zeros_like(reconstruction) 
            # with torch.no_grad():
            #     for i in range(len(reconstruction)):
            #         data_i = reconstruction[i, 0, :].cpu().numpy()  
            #         data_i = zero_from_first_chunk_with_peaks(data_i, zero_threshold=0.05, min_zero_length=5, peak_threshold=0.1)
            #         data_i = moving_average_filter(data_i, window_size=9)
            #         smoothed_reconstruction[i, 0, :] = torch.tensor(data_i).to(device)
        
            # Compute loss 
            # KL loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            # Recon loss
            recon_loss = criterion(reconstruction, data)
            loss = recon_loss + alpha * kl_loss
            
            # Record loss
            running_loss += loss.item() * len(params)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

class WeightedMSELoss(nn.Module):
    # Calculate the weighted loss for each batch, then average it
    def __init__(self, class_weights) -> None:
        super(WeightedMSELoss, self).__init__()
        self.class_weights = class_weights
        self.MSE = nn.MSELoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor, d_type: torch.Tensor) -> torch.Tensor:
        # d_type: type of the ground truth signal
        # target: ground truth signal
        # input: predicted signal
        weights = torch.tensor([self.class_weights[int(label.item())] for label in d_type], dtype=torch.float32, device=target.device)
        mse = self.MSE(pred, target).mean(dim=(1, 2))  # Compute element-wise MSE
        weighted_mse = mse * weights  # Apply weights
        weighted_mse = weighted_mse.mean() # take the average
        
        return weighted_mse