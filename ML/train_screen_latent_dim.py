import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from torch.utils.data import DataLoader
from resources.plot_utils import plot_R2
from VAE_core import VAE, train_VAE, validate_VAE, test_VAE
from MLP_VAE_core import CustomDataset, CombinedModel, train_combined, validate_combined, test_combined, count_parameters

def scale_feature(value, min_val, max_val, option):
    if option == "linear":
        return (value - min_val) / (max_val - min_val)
    elif option == "exp":
        return (np.log(value) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
    else:
        raise ValueError(f"Unknown scaling option: {option}")

def scale_dataset(dataset, ranges, opt):
    scaled_dataset = np.zeros_like(dataset)
    for i, (key, range_vals) in enumerate(ranges.items()):
        min_val, max_val = range_vals
        scaling_option = opt[i]
        scaled_dataset[:, i] = [scale_feature(val, min_val, max_val, scaling_option) for val in dataset[:, i]]
    return scaled_dataset

    
def main(config_file):

    # Load config file
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(config)
    
    # Set paths
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

    # Load RFP profiles
    data_array = np.load(os.path.join(data_dir, config["dataset"]["outputs_filename"]))
    data_array = data_array.reshape([-1, 3, 201])
    RFP_data = data_array[:, 1, :].squeeze()
    normalized_RFP = RFP_data / RFP_data.max(axis=1, keepdims=True)
    normalized_RFP = normalized_RFP.reshape([-1, 1, 201])

    # Parameters 
    data_array = np.load(os.path.join(data_dir, config["dataset"]["params_filename"])) # original scale
    scaling_ranges = config['dataset']['scaling_ranges']
    scaling_options = config['dataset']['scaling_options']
    all_params = config['dataset']['all_params']
    sceening_params = config['dataset']['sceening_params']
    selected_param_idx = [all_params.index(param) for param in sceening_params]
    params_array = params_array[:, selected_param_idx]

    # Pattern types
    pattern_types_array = np.load(os.path.join(data_dir, config["dataset"]["types_filename"]))
    pattern_types_array = pattern_types_array[:, 1]
    
    print('---------------------------------------------')
    print(f"RFP profiles: {normalized_RFP.shape}")
    print(f"Parameters: {params_array.shape}")
    print(f"Pattern types:  {pattern_types_array.shape}")

    # Get device
    device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Hyperparameters
    batch_size = config["training_parameters"]["batch_size"]
    seq_length = data_array.shape[2]
    latent_dim_list = config["model_architecture"]["latent_dim_list"]
    latent_channel = config["model_architecture"]["latent_channel"]
    num_models = len(latent_dim_list)

    vae_r2_traing_list = []
    vae_r2_test_list = []
    mlp_r2_traing_list = []
    mlp_r2_test_list = []

    # Split test set
    test_split = config["dataset"]["test_split"]
    random_state = config["dataset"]["random_state"]
    data_array = torch.tensor(data_array).float()
    train_data_all, test_data, _, _ = train_test_split(normalized_RFP, range(normalized_RFP.shape[0]), test_size=test_split, random_state=random_state)
    train_params_all, test_params, _, _ = train_test_split(params_array, range(params_array.shape[0]), test_size=test_split, random_state=random_state)
    train_labels_all, test_labels, _, _ = train_test_split(pattern_types_array, range(pattern_types_array.shape[0]), test_size=test_split, random_state=random_state)

    test_data = torch.tensor(test_data[:, 0, :].reshape([-1, 1, seq_length])).float()
    train_data_all = torch.tensor(train_data_all[:, 0, :].reshape([-1, 1, seq_length])).float()

    # Loop for different latent dimensions
    for i in range(0, num_models):
        
        latent_dim = latent_dim_list[i]

        print(f' ---------------------- latent dim: {latent_dim} ---------------------- ', end='\\')
        
        # Split train and validation datasets
        validation_split = config["dataset"]["validation_split"]
        train_data, valid_data, _, _ = train_test_split(train_data_all, range(train_data_all.shape[0]), test_size=validation_split, random_state=random_state)
        train_params, valid_params, _, _ = train_test_split(train_params_all, range(train_params_all.shape[0]), test_size=validation_split, random_state=random_state)
        train_labels, valid_labels, _, _ = train_test_split(train_labels_all, range(train_labels_all.shape[0]), test_size=validation_split, random_state=random_state)

        # Normalize parameters
        train_params = scale_dataset(train_params, scaling_ranges, scaling_options)
        valid_params = scale_dataset(valid_params, scaling_ranges, scaling_options)
        test_params = scale_dataset(test_params, scaling_ranges, scaling_options)

        # Dataloader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        # Check dataset size
        print('Train set size: ', len(train_data))
        print('Valid set size: ', len(valid_data))
        print('Test set size: ', len(test_data))

        print('------------------- VAE training ---------------------------', end='\\')

        # Initiate VAE 
        vae = VAE(seq_length, latent_dim, latent_channel)
        vae = vae.to(device)

        # Training setup
        epochs = config["training_parameters"]["epochs_VAE"]
        lr= config["training_parameters"]["learning_rate_VAE"]   
        min_lr = config['training_parameters']['min_lr_VAE']     
        gamma = config['training_parameters']['gamma_VAE'] 
        weight_decay = config['training_parameters']['weight_decay_VAE'] 
        alpha = config['training_parameters']['alpha_VAE'] 
        criterion = MSELoss()
        optimizer = Adam(vae.parameters(), lr=lr)

        # Early stopping
        best_valid_loss = np.inf
        epochs_no_improve = 0
        patience = config["training_parameters"]["patience_VAE"]

        # Warm up
        warmup_epochs_VAE = config['training_parameters']['warmup_epochs_VAE']
        
        def warmup_scheduler_VAE(epoch, warmup_epochs):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 1.0
    
        scheduler1 = LambdaLR(optimizer, lr_lambda=warmup_scheduler_VAE)
        scheduler2 = ExponentialLR(optimizer, gamma=gamma)

        # Train 
        train_loss_history = []
        valid_loss_history = []
        test_loss_history = []

        for epoch in range(config["training_parameters"]["epochs"]):
            train_loss = train_VAE(vae, train_loader, optimizer, criterion, alpha, device)
            valid_loss = validate_VAE(vae, valid_loader, criterion, alpha, device)
            test_loss = test_VAE(vae, test_loader, criterion, alpha, device)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            test_loss_history.append(test_loss)

            # Print loss
            if (epoch) % 5 == 0: # every 5 epochs
                print('Epoch: {} Train: {:.7f}, Valid: {:.7f}, Test: {:.7f}, Lr:{:.8f}'.format(epoch + 1, train_loss_history[epoch], valid_loss_history[epoch], test_loss_history[epoch], param_group['lr']))

            # Clamp minimum learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], min_lr)

            # Update learning rate
            if epoch < warmup_epochs_VAE:
                scheduler1.step()
            else:
                scheduler2.step()
            scheduler2.step()

            # Check for early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0
                torch.save(vae.state_dict(), os.path.join(log_dir, config['model_name']['VAE_model_prefix'] + str(latent_dim) + '.pt'))
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience_VAE:
                print('Early stopping!', end='\\')
                break  # Exit the loop
    
        # Save VAE
        model_path = os.path.join(model_dir, config['model_name']['VAE_model_prefix'] + str(latent_dim) + '.pt')
        print('model path: ', model_path)
        torch.save(vae.state_dict(), model_path)
        
        print('------------------- Evaluate VAE performance ---------------------------', end='\\')
        # Calculate trained VAE accuracy
        train_data = train_data.cpu()
        train_data_ori = torch.tensor(train_data[0:1000], dtype=torch.float32).to(device)
        test_data_ori = torch.tensor(test_data[0:1000], dtype=torch.float32).to(device)
        with torch.no_grad():
            train_pred, _, _ = vae(train_data_ori)
            test_pred, _, _ = vae(test_data_ori)

        # Squeeze the output to match the original data dimension
        train_pred = train_pred.squeeze(1).cpu().numpy()
        test_pred = test_pred.squeeze(1).cpu().numpy()
        train_data_ori = train_data_ori.cpu().numpy().squeeze(1)
        test_data_ori = test_data_ori.cpu().numpy().squeeze(1)

        r2_train = r2_score(train_data_ori.flatten(), train_pred.flatten())
        r2_test = r2_score(test_data_ori.flatten(), test_pred.flatten())
        
        print('******** VAE ********')
        print('R2 train: ', r2_train)
        print('R2 test: ', r2_test)
        vae_r2_traing_list.append(r2_train)
        vae_r2_test_list.append(r2_test)

        print('------------------- MLP training ---------------------------')
        # Create datasets
        train_dataset = CustomDataset_with_type(torch.tensor(train_params, dtype=torch.float32), 
                                torch.tensor(train_data, dtype=torch.float32),
                                torch.tensor(train_labels, dtype=torch.float32))
        valid_dataset = CustomDataset_with_type(torch.tensor(valid_params, dtype=torch.float32), 
                                    torch.tensor(valid_data, dtype=torch.float32),
                                    torch.tensor(valid_labels, dtype=torch.float32))
        test_dataset = CustomDataset_with_type(torch.tensor(test_params, dtype=torch.float32), 
                                    torch.tensor(test_data, dtype=torch.float32),
                                    torch.tensor(test_labels, dtype=torch.float32))
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        print('Traning data points:', len(train_dataset))
        print('Valid data points:', len(valid_dataset))
        print('Test data points:', len(test_dataset))

        # Initiate MLP
        input_dim = len(train_params[0,:])
        model = CombinedModel(input_dim, latent_dim, vae.decoder, 42)
        
        # # Load model if exist
        if config['model_name']['combined_load_pretrained']:
            model_path = os.path.join(model_dir, config['model_name']['combined_model_prefix'] + str(latent_dim) + '.pt')
            print(model_path)
            model.load_state_dict(torch.load(model_path))

        model = model.to(device)

        # Training setup
        epochs = config['training_parameters']['epochs_combined'] 
        lr = config['training_parameters']['learning_rate_combined'] 
        min_lr = config['training_parameters']['min_lr_combined']     
        gamma = config['training_parameters']['gamma_combined'] 
        weight_decay = config['training_parameters']['weight_decay_combined'] 
        alpha = config['training_parameters']['alpha_combined'] 
        patience_combined = config['training_parameters']['patience_combined']

        criterion = MSELoss()
        optimizer = Adam(model.mlp.parameters(), lr=lr, weight_decay=weight_decay)  # Only train MLP parameters

        #  Warmup 
        warmup_epochs_combined = config['training_parameters']['warmup_epochs_combined']
        def warmup_scheduler_combined(epoch, warmup_epochs):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 1.0
        # Scheduler 
        scheduler1 = LambdaLR(optimizer, lr_lambda=warmup_scheduler_combined)
        scheduler2 = ExponentialLR(optimizer, gamma=0.995)

        # Early stopping setup
        best_valid_loss = np.inf
        epochs_no_improve = 0
        patience = config['training_parameters']['patience_combined']
        
        # Training loop
        train_loss_history = []
        valid_loss_history = []
        test_loss_history = []

        for epoch in range(epochs):

            train_loss = train_combined(model, train_loader, optimizer, criterion, alpha, device)
            valid_loss = validate_combined(model, valid_loader, criterion, alpha, device)
            test_loss = test_combined(model, test_loader, criterion, alpha, device)

            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            test_loss_history.append(test_loss)

            # Clamp minimum learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], min_lr)

            # Print loss
            if epoch % 5 == 0: # every 5 epochs
                print('Epoch: {} Train: {:.7f}, Valid: {:.7f}, Test: {:.7f}, Lr:{:.8f}'.format(epoch + 1, train_loss_history[epoch], valid_loss_history[epoch], test_loss_history[epoch], param_group['lr']))

            # Update learning rate
            if epoch < warmup_epochs_combined:
                scheduler1.step()
            else:
                scheduler2.step()

            # Check for early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0  # Reset the counter
            else:
                epochs_no_improve += 1  # Increment the counter

            if epochs_no_improve == patience_combined:
                print('Early stopping!')
                break  # Exit the loop
            if valid_loss < 0.0001:
                break

        # Save MLP
        model_path = os.path.join(log_dir, config['model_name']['combined_model_prefix'] + str(latent_dim) + '.pt')
        print('model path: ', model_path)
        torch.save(model.state_dict(), model_path)
        
        print('------------------- Evaluate MLP - VAE performance ---------------------------')
        # Calculate trained MLP accuracies
        train_pred = []
        train_ori = []
        test_pred = []
        test_ori = []

        model.eval()
        with torch.no_grad():
            for params, data, _ in train_loader:
                data = data.to(device)
                params = params.to(device)
                reconstruction, mean, logvar = model(params)
                train_pred.append(reconstruction.cpu().numpy())
                train_ori.append(data.cpu().numpy())

            for params, data, _ in test_loader:
                data = data.to(device)
                params = params.to(device)
                reconstruction, mean, logvar = model(params)
                test_pred.append(reconstruction.cpu().numpy())
                test_ori.append(data.cpu().numpy())

        # Concatenate
        train_pred = np.concatenate(train_pred)
        train_ori = np.concatenate(train_ori)
        test_pred = np.concatenate(test_pred)
        test_ori = np.concatenate(test_ori)
        
        filename = log_dir + config['model_name']['combined_model_prefix'] + 'train_R2.png'
        plot_R2(train_ori, train_pred, filename)
        filename = log_dir + config['model_name']['combined_model_prefix'] + 'test_R2.png'
        plot_R2(test_ori, test_pred, filename)
        
        # Squeeze the output to match the original data dimension
        train_pred = train_pred.squeeze(1)
        test_pred = test_pred.squeeze(1)
        train_ori = train_ori.squeeze(1)
        test_ori = test_ori.squeeze(1)

        r2_train = r2_score(train_ori.flatten(), train_pred.flatten())
        r2_test = r2_score(test_ori.flatten(), test_pred.flatten())
        
        print('******** MLP ********')
        print('R2 train: ', r2_train)
        print('R2 test: ', r2_test)
        mlp_r2_traing_list.append(r2_train)
        mlp_r2_test_list.append(r2_test)
        
    # Plot accuracy vs latent dim
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    axs[0].plot(latent_dim_list, vae_r2_test_list)
    axs[1].plot(latent_dim_list, mlp_r2_test_list)
    axs[0].set_title('VAE')
    axs[1].set_title('MLP - VAE')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'test accuracy vs latent dim.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Screen MLP-VAE models latent dimension.')
    parser.add_argument('config', type=str, help='JSON configuration filename')
    args = parser.parse_args()
    main(args.config)