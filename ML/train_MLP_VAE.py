import os
import json
import argparse
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from resources.data_utils import peak_to_ring_num, scale_dataset, get_RFP_type 
from resources.plot_utils import plot_R2, plot_profiles_peak_1channel, plot_training_history
from MLP_VAE_core import CustomDataset, CombinedModel, train_combined, validate_combined, test_combined, count_parameters
from VAE_core import VAE 

def warmup_scheduler(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        return 1.0
    
def R2_acc_by_type(ori_types, pred_types, ori_data, pred_data, target_class):

    # Compute R2
    indices = [i for i, t in enumerate(ori_types) if t == target_class]

    ori_data_selected = ori_data[indices]
    pred_data_selected = pred_data[indices]
    
    if len(ori_data_selected) != 0:
        ori_data_selected = ori_data_selected.flatten()
        pred_data_selected = pred_data_selected.flatten()

        r2 = r2_score(ori_data_selected, pred_data_selected)
    else:
        r2 = 0
        
    correct_predictions = sum([1 for i in indices if ori_types[i] == pred_types[i]])
    total_predictions = len(indices)
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
    else:
        accuracy = 0

    return r2, accuracy, correct_predictions, total_predictions


def main(json_file):

    # Load the configuration from the provided JSON file
    with open(json_file, 'r') as f:
        config = json.load(f)
    print(config)
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Set paths from the JSON config
    data_dir = config['paths']['data_dir']
    model_dir = config['paths']['model_dir']
    log_dir = config['paths']['log_dir']
    checkpoint_dir = config['checkpointing']['checkpoint_dir']
    log_dir = config['paths']['log_dir']
    vae_model_dir = config['paths']['vae_model_dir']

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = f"{current_time}"
    
    model_dir = os.path.join(log_dir, model_dir)
    log_dir = os.path.join(log_dir, foldername)
    checkpoint_dir = os.path.join(checkpoint_dir, foldername)
    
    # Create the new folder
    os.makedirs(model_dir, exist_ok=True) 
    os.makedirs(log_dir, exist_ok=True) 
    os.makedirs(checkpoint_dir, exist_ok=True) 

    # Logging
    shutil.copy(json_file, log_dir)

    ## -------------------- Read in training data --------------------
    # RFP profiles
    data_array =  np.load(os.path.join(data_dir, config["dataset"]["outputs_filename"]))
    data_array = data_array.reshape([-1, 3, 201])
    RFP_data = data_array[:, 1, :].squeeze()
    print(f"RFP profiles: {RFP_data.shape}")

    # Normalize RFP profiles
    normalized_RFP = RFP_data / RFP_data.max(axis=1, keepdims=True)
    normalized_RFP = normalized_RFP.reshape([-1, 1, 201])
    print(f"Normalized RFP profiles: {normalized_RFP.shape}")

    # Parameters 
    params_array = np.load(os.path.join(data_dir + config["dataset"]["params_filename"]))
    print(f"params_array: {params_array.shape}")
    scaling_ranges = config["dataset"]["scaling_ranges"]
    scaling_options = config["dataset"]["scaling_options"]
    all_params = config["dataset"]["all_params"]
    sceening_params = config["dataset"]["selected_params"]
    selected_param_idx = [all_params.index(param) for param in sceening_params]
#     params_array = params_array[:, selected_param_idx]
#     print(f"params_array: {params_array.shape}")

    # Pattern types
    pattern_types_array = np.load(os.path.join(data_dir, config["dataset"]["types_filename"]), allow_pickle=True)
    RFP_types_array = pattern_types_array[:, 1]

    # Check size
    print('---------------------------------------------')
    print(f"RFP profiles: {normalized_RFP.shape}")
    print(f"Parameters: {params_array.shape}")
    print(f"Pattern types:  {RFP_types_array.shape}")

    ## ---------------- Dataloader ----------------
    # Set up hyperparameters
    batch_size = config['training_parameters']['batch_size']
    latent_dim = config['model_architecture']['latent_dim']
    latent_channel = config['model_architecture']['latent_channel']
    seq_length = normalized_RFP.shape[2] 
    input_dim = params_array.shape[1]

    # Set everything to float32
    normalized_RFP = normalized_RFP.astype(np.float32)
    params_array = params_array.astype(np.float32)

    # Shuffle
    normalized_RFP = shuffle(normalized_RFP, random_state=config['dataset']['random_state'])
    params_array = shuffle(params_array, random_state=config['dataset']['random_state'])
    RFP_types_array = shuffle(RFP_types_array, random_state=config['dataset']['random_state'])

    # Split datasets
    train_data, test_data, _, _ = train_test_split(normalized_RFP, range(len(normalized_RFP)), test_size=config['dataset']['test_split'], random_state=config['dataset']['random_state'], shuffle=False)
    train_params, test_params, _, _ = train_test_split(params_array, range(len(params_array)), test_size=config['dataset']['test_split'], random_state=config['dataset']['random_state'], shuffle=False)
    train_types, test_types, _, _ = train_test_split(RFP_types_array, range(len(RFP_types_array)), test_size=config['dataset']['test_split'], random_state=config['dataset']['random_state'], shuffle=False)

    train_data, valid_data, _, _ = train_test_split(train_data, range(len(train_data)), test_size=config['dataset']['validation_split'], random_state=config['dataset']['random_state'], shuffle=False)
    train_params, valid_params, _, _ = train_test_split(train_params, range(len(train_params)), test_size=config['dataset']['validation_split'], random_state=config['dataset']['random_state'], shuffle=False)
    train_types, valid_types, _, _ = train_test_split(train_types, range(len(train_types)), test_size=config['dataset']['validation_split'], random_state=config['dataset']['random_state'], shuffle=False)

    # Inputs
    norm_train_params = scale_dataset(train_params, scaling_ranges, scaling_options)
    norm_valid_params = scale_dataset(valid_params, scaling_ranges, scaling_options)
    norm_test_params = scale_dataset(test_params, scaling_ranges, scaling_options)

    # Dataset
    train_dataset = CustomDataset(norm_train_params, train_data, train_types)
    valid_dataset = CustomDataset(norm_valid_params, valid_data, valid_types)
    test_dataset = CustomDataset(norm_test_params, test_data, test_types)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(' -------------------- Train set -------------------- ')
    print('Data: ', train_data.shape)
    print('Parameters: ', train_params.shape)
    print('Pattern type: ', train_types.shape)

    print(' -------------------- Validation set -------------------- ')
    print('Data: ', valid_data.shape)
    print('Parameters: ', valid_params.shape)
    print('Pattern type: ', valid_types.shape)

    print(' -------------------- Test set -------------------- ')
    print('Data: ', test_data.shape)
    print('Parameters: ', test_params.shape)
    print('Pattern type: ', test_types.shape)

    print(' -------------------- Check params ranges -------------------- ')
    # check parameter ranges
    upper_lims = np.max(norm_train_params, axis=0)
    lower_lims = np.min(norm_train_params, axis=0)
    for i in range(len(upper_lims)):
        print(sceening_params[i], ' -- Upper limits -- ',upper_lims[i], 'Lower limits -- ',lower_lims[i])

    # Load trained VAE
    vae = VAE(seq_length, latent_dim, latent_channel)
    vae.load_state_dict(torch.load(os.path.join(model_dir, config['model_name']['VAE_model_name'])))
    vae = vae.to(device)
    vae.eval() 

    ## ---------------- Train ----------------
    # Initiate model
    model = CombinedModel(input_dim, latent_dim, vae.decoder, 42)

    # # Load trained model if exist
    if config['model_name']['exist_pre_trained']:
        filename = model_dir + config['model_name']['pre_trained_combined_model_name']
        model.load_state_dict(torch.load(filename))

    # Freeze VAE parameters
    for param in model.decoder.parameters():
        param.requires_grad = False
    print(model)
    model = model.to(device)
    print("The model has", count_parameters(model), "trainable parameters")

    # Training setup
    min_lr = config['training_parameters']['min_lr']  
    epochs = config['training_parameters']['epochs']
    gamma = config['training_parameters']['gamma']
    weight_decay = config['training_parameters']['weight_decay']
    alpha = config['training_parameters']['alpha']
    lr = config['training_parameters']['learning_rate']
    epochs = config['training_parameters']['epochs']
    optimizer_name = config['training_parameters']['optimizer']
    if_early_stopping = config['training_parameters']['early_stopping']

    # Training setup
    criterion = nn.MSELoss()
    optimizer_dict = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
    }
    optimizer = optimizer_dict[optimizer_name](model.mlp.parameters(), lr=lr, weight_decay=weight_decay)


    # Early stopping 
    best_valid_loss = np.inf  
    epochs_no_improve = 0  
    patience = config['training_parameters']['patience'] 

    # Warm up
    warmup_epochs = config['training_parameters']['warmup_epochs'] 
    def warmup_scheduler(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0
        
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_scheduler)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Training loop
    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []

    for epoch in tqdm(range(epochs)):

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

        # Check for early stopping
        if if_early_stopping:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0  # Reset the counter
            else:
                epochs_no_improve += 1  # Increment the counter

            if epochs_no_improve == patience:
                print('Early stopping!')
                break  # Exit the loop

    # Plotting the loss history
    filename = os.path.join(log_dir, 'train_history.png')
    plot_training_history(filename, train_loss_history, valid_loss_history, test_loss_history)

    ## Save
    model_path = os.path.join(model_dir, config['model_name']['combined_model_name'])
    torch.save(model.state_dict(), model_path)

    ## Traing results
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
    train_pred = np.concatenate(train_pred)#.flatten()
    train_ori = np.concatenate(train_ori)#.flatten()
    test_pred = np.concatenate(test_pred)#.flatten()
    test_ori = np.concatenate(test_ori)#.flatten()

    filename = log_dir + 'VAE_train_R2.png'
    plot_R2(train_ori, train_pred, filename)
    filename = log_dir + 'VAE_test_R2.png'
    plot_R2(test_ori, test_pred, filename)

    ## ---------------- Accuracy by class ----------------
    print(' ------------------------- Train ------------------------- ')
    _, ori_types = get_RFP_type(train_ori)
    _, pred_types = get_RFP_type(train_pred)
    ori_types = peak_to_ring_num(np.array(ori_types))
    pred_types = peak_to_ring_num(np.array(pred_types))

    # Get R2 and acc for each pattern type
    R2_1, acc_1, correct_1, total_1 = R2_acc_by_type(ori_types, pred_types, train_ori, train_pred, 1)
    R2_2, acc_2, correct_2, total_2 = R2_acc_by_type(ori_types, pred_types, train_ori, train_pred, 2)
    R2_3, acc_3, correct_3, total_3 = R2_acc_by_type(ori_types, pred_types, train_ori, train_pred, 3)
    R2_4, acc_4, correct_4, total_4 = R2_acc_by_type(ori_types, pred_types, train_ori, train_pred, 4)

    print(' 1 ring --- R2: ', R2_1, ' , acc: ', acc_1 , ', correct#: ', correct_1, ', total#: ', total_1)
    print(' 2 ring --- R2: ', R2_2, ' , acc: ', acc_2 , ', correct#: ', correct_2, ', total#: ', total_2)
    print(' 3 ring --- R2: ', R2_3, ' , acc: ', acc_3 , ', correct#: ', correct_3, ', total#: ', total_3)
    print(' 4 ring --- R2: ', R2_4, ' , acc: ', acc_4 , ', correct#: ', correct_4, ', total#: ', total_4)

    print(' ------------------------- Test ------------------------- ')
    _, ori_types = get_RFP_type(test_ori)
    _, pred_types = get_RFP_type(test_pred)
    ori_types = peak_to_ring_num(np.array(ori_types))
    pred_types = peak_to_ring_num(np.array(pred_types))

    # Get R2 and acc for each pattern type
    R2_1, acc_1, correct_1, total_1 = R2_acc_by_type(ori_types, pred_types, test_ori, test_pred, 1)
    R2_2, acc_2, correct_2, total_2 = R2_acc_by_type(ori_types, pred_types, test_ori, test_pred, 2)
    R2_3, acc_3, correct_3, total_3 = R2_acc_by_type(ori_types, pred_types, test_ori, test_pred, 3)
    R2_4, acc_4, correct_4, total_4 = R2_acc_by_type(ori_types, pred_types, test_ori, test_pred, 4)

    print(' 1 ring --- R2: ', R2_1, ' , acc: ', acc_1 , ', correct#: ', correct_1, ', total#: ', total_1)
    print(' 2 ring --- R2: ', R2_2, ' , acc: ', acc_2 , ', correct#: ', correct_2, ', total#: ', total_2)
    print(' 3 ring --- R2: ', R2_3, ' , acc: ', acc_3 , ', correct#: ', correct_3, ', total#: ', total_3)
    print(' 4 ring --- R2: ', R2_4, ' , acc: ', acc_4 , ', correct#: ', correct_4, ', total#: ', total_4)
        
    output = (
    f"1 ring --- R2: {R2_1}, acc: {acc_1}, correct#: {correct_1}, total#: {total_1}\n"
    f"2 ring --- R2: {R2_2}, acc: {acc_2}, correct#: {correct_2}, total#: {total_2}\n"
    f"3 ring --- R2: {R2_3}, acc: {acc_3}, correct#: {correct_3}, total#: {total_3}\n"
    f"4 ring --- R2: {R2_4}, acc: {acc_4}, correct#: {correct_4}, total#: {total_4}\n"
    )

    # Write the output to a file
    with open(os.path.join(log_dir,'results.txt'), 'w') as f:
        f.write(output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the MLP-VAE from end to end.')
    parser.add_argument('config', type=str, help='JSON configuration filename')
    args = parser.parse_args()
    main(args.config)

# sbatch -p youlab-gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --wrap="python3 -u train_MLP_VAE.py config_train_MLP_VAE.json"
