import os
import json
import argparse
import shutil
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from resources.plot_utils import plot_R2, plot_training_history
from resources.checkpoints_utils import save_checkpoint
from VAE_core import VAE, train_VAE, validate_VAE, test_VAE, count_parameters  # Import from your VAE file


def main(json_file):

    # Load the configuration from the provided JSON file
    with open(json_file, 'r') as f:
        config = json.load(f)
    print(config,flush=True)
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print('Device: ', device,flush=True)
    
    # Set paths from the JSON config
    data_dir = config['paths']['data_dir']
    model_dir = config['paths']['model_dir']
    log_dir = config['paths']['log_dir']
    checkpoint_dir = config['checkpointing']['checkpoint_dir']
    
    # Get current time, use as folder name for this training
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = f"{current_time}"
    
    model_dir = os.path.join(log_dir, model_dir)
    log_dir = os.path.join(log_dir, foldername)
    checkpoint_dir = os.path.join(checkpoint_dir, foldername)
    
    # Create the new folder
    os.makedirs(model_dir, exist_ok=True) 
    os.makedirs(log_dir, exist_ok=True) 
    os.makedirs(checkpoint_dir, exist_ok=True) 

    # Copy config file to folder for logging
    shutil.copy(json_file, log_dir)
   
    # RFP profiles
    filename = os.path.join(data_dir, config["dataset"]["outputs_filename"])
    data_array = np.load(filename)
    data_array = data_array.reshape([-1, 3, 201])
    RFP_data = data_array[:, 1, :].squeeze()
    print(f"RFP profiles: {RFP_data.shape}")

    # Normalize RFP profiles
    normalized_RFP = RFP_data / RFP_data.max(axis=1, keepdims=True)
    print(f"Normalized RFP profiles: {normalized_RFP.shape}")

    # Plot -- Create 100 panels (10x10), each showing a random data series
    if config['dataset']['plot_examples'] == True:
        rows = 10
        cols = 10
        fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
        fig.suptitle('Random RFP Data', fontsize=16)
        for i in range(rows):
            for j in range(cols):
                random_index = np.random.randint(normalized_RFP.shape[0])
                axs[i, j].plot(normalized_RFP[random_index])
                axs[i, j].set_ylim([0, 1])
                axs[i, j].axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        figname = os.path.join(log_dir, 'data.png')
        fig.savefig(figname)  

    # Model and training parameters from JSON config
    batch_size = config['training_parameters']['batch_size']
    latent_dim = config['model_architecture']['latent_dim']
    latent_channel = config['model_architecture']['latent_channel']
    alpha = config['training_parameters']['alpha'] 
    lr = config['training_parameters']['learning_rate']
    min_lr = config['training_parameters']['min_lr']
    epochs = config['training_parameters']['epochs']
    gamma = config['training_parameters']['gamma']
    weight_decay = config['training_parameters']['weight_decay']
    optimizer_name = config['training_parameters']['optimizer']
    if_early_stopping = config['training_parameters']['early_stopping']

    # Split data
    data = normalized_RFP
    data = torch.tensor(data).float().unsqueeze(1)
    train_data, test_data, _, _ = train_test_split(data, 
                                                    range(len(data)), 
                                                    test_size=config['dataset']['test_split'], 
                                                    random_state=config['dataset']['random_state'], 
                                                    shuffle=False)
    train_data, valid_data, _, _ = train_test_split(train_data, 
                                                    range(len(train_data)), 
                                                    test_size=config['dataset']['validation_split'], 
                                                    random_state=config['dataset']['random_state'], 
                                                    shuffle=True)

    print('Train data size: ', train_data.shape)
    print('Validation data size: ', valid_data.shape)
    print('Test data size: ', test_data.shape)

    # Prepare DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Initiate model
    model = VAE(data.shape[2], latent_dim, latent_channel)
    print(model)
    model = model.to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer_dict = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
    }
    optimizer = optimizer_dict[optimizer_name](model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping
    best_test_loss = np.inf  
    epochs_no_improve = 0  # Counter for epochs since the test loss last improved
    patience = config['training_parameters']['patience']  # Patience for early stopping

    # Learning rate schedulers
    warmup_epochs = config['training_parameters']['warmup_epochs']
    def warmup_scheduler(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_scheduler)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Training loop
    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []

    for epoch in tqdm(range(epochs)):
        train_loss = train_VAE(model, train_loader, optimizer, criterion, alpha, device)
        valid_loss = validate_VAE(model, valid_loader, criterion, alpha, device)
        test_loss = test_VAE(model, test_loader, criterion, alpha, device)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        test_loss_history.append(test_loss)

         # Clamp minimum learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)

         # Print loss
        if (epoch + 1) % 5 == 0: # every 5 epochs
            print('Epoch: {} Train: {:.7f}, Valid: {:.7f}, Test: {:.7f}, Lr:{:.8f}'.format(epoch + 1, train_loss_history[epoch], valid_loss_history[epoch], test_loss_history[epoch], param_group['lr']))
    
        # Save checkpoint periodically
        if (epoch+1) % config['checkpointing']['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, test_loss, checkpoint_dir)
        
        # Update learning rate
        if epoch < warmup_epochs:
            scheduler1.step()
        else:
            scheduler2.step()

        # Early stopping check
        if if_early_stopping:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print('Early stopping!')
                break

    # Plot and save training history
    filename = os.path.join(log_dir, 'train_history.png')
    plot_training_history(filename, train_loss_history, valid_loss_history, test_loss_history)

    # Plot R2
    model.eval()
    train_data_short = train_data[0: len(test_data)]
    test_data_short = test_data

    with torch.no_grad():
        train_data_short = train_data_short.to(device)
        test_data_short = test_data_short.to(device)

        train_pred, _, _= model(train_data_short)
        test_pred, _, _ = model(test_data_short)
        
    train_data_short = train_data_short.squeeze(1).cpu().numpy()
    test_data_short = test_data_short.squeeze(1).cpu().numpy()
    train_pred = train_pred.squeeze(1).cpu().numpy()
    test_pred = test_pred.squeeze(1).cpu().numpy()
    filename = log_dir + 'VAE_train_R2.png'
    plot_R2(train_data_short, train_pred, filename)
    filename = log_dir + 'VAE_test_R2.png'
    plot_R2(test_data_short, test_pred, filename)

    # Plot examples
    if config["performance"]['plot_examples']:
        fig, axs = plt.subplots(2, 5, figsize=(10, 3))
        for i in range(5):
            axs[0, i].plot(train_data_short[i].squeeze(), label='Original', color='blue')
            axs[0, i].plot(train_pred[i], label='Reconstructed', color='orange')
            axs[0, i].set_title(f'Train {i + 1}')
            # axs[0, i].legend()

        for i in range(5):
            axs[1, i].plot(test_data_short[i].squeeze(), label='Original', color='blue')
            axs[1, i].plot(test_pred[i], label='Reconstructed', color='orange')
            axs[1, i].set_title(f'Test {i + 1}')
            # axs[1, i].legend()
        plt.tight_layout()
        figname = os.path.join(log_dir, 'rand_VAE_preds.png')
        fig.savefig(figname)

    # Save final model
    final_model_path = os.path.join(model_dir, config['model_name'])
    torch.save(model.state_dict(), final_model_path)
    print(f'Model saved at {final_model_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE model.')
    parser.add_argument('config', type=str, help='JSON configuration filename')
    args = parser.parse_args()
    main(args.config)

# sbatch -p youlab-gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --wrap="python3 -u train_VAE.py config_train_VAE.json"
