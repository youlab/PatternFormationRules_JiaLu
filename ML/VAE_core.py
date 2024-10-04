import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, seq_length, latent_dim, latent_channel):
        super(CNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length

        if latent_dim == 2:
             self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(8, latent_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
                
        elif latent_dim == 4:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, latent_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        elif latent_dim == 8:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, latent_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        elif latent_dim == 16:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, latent_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
             
            
        elif latent_dim == 32:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, latent_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        elif latent_dim == 64:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, latent_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

        self.fc_mean = nn.Linear(latent_channel * seq_length, latent_dim)
        self.fc_logvar = nn.Linear(latent_channel * seq_length, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

class CNNDecoder(nn.Module):
    def __init__(self, seq_length, latent_dim, latent_channel):
        super(CNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length
        
        self.fc = nn.Linear(latent_dim, latent_channel * seq_length)

        if latent_dim == 2:
            self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose1d(latent_channel, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(8, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
            )
        elif latent_dim == 4:
            self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose1d(latent_channel, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
            )
        elif latent_dim == 8:
            self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose1d(latent_channel, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
            )
        elif latent_dim == 16:
            self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose1d(latent_channel, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
            )
            
        elif latent_dim == 32:
            self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose1d(latent_channel, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
            )
        elif latent_dim == 64:
            self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose1d(latent_channel, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
            )
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.latent_channel, self.seq_length)
        x = self.decoder(x)
        return F.relu(x)
    
class VAE(nn.Module):
    def __init__(self, seq_length, latent_dim, latent_channel):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length

        self.encoder = CNNEncoder(seq_length, latent_dim, latent_channel)
        self.decoder = CNNDecoder(seq_length, latent_dim, latent_channel)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mean, logvar

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_VAE(model, dataloader, optimizer, criterion, alpha, device):
    
    model.train()
    running_loss = 0
    
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mean, logvar = model(data)
        
        # Compute loss - MSE + alpha * KL-divergence
        recon_loss = criterion(reconstruction, data)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + alpha * kl_loss
 
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_VAE(model, dataloader, criterion, alpha, device):

    model.eval()
    running_loss = 0

    with torch.no_grad():
        for data in dataloader:
            
            data = data.to(device)
            
            reconstruction, mean, logvar = model(data)

            # Compute loss - MSE + alpha * KL-divergence
            recon_loss = criterion(reconstruction, data)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + alpha * kl_loss

            running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss

def test_VAE(model, dataloader, criterion, alpha, device):
    
    model.eval()
    running_loss = 0
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            
            reconstruction, mean, logvar = model(data)
            
            # Compute loss - MSE + alpha * KL-divergence            
            recon_loss = criterion(reconstruction, data)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + alpha * kl_loss

            running_loss += loss.item() * data.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

