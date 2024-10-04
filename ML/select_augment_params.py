import os
import numpy as np
import torch
import json
import argparse

def expand_to_num(types_array, params_array, total_num):
    
    unique_values, counts = np.unique(types_array, return_counts=True)
    expanded_array = []
    expanded_params_array = []
    
    for value, count in zip(unique_values, counts):
        repeat_times = round(total_num/count)
        indices = np.where(types_array == value)
        
        # extend type array
        new = np.repeat(types_array[indices[0]], repeat_times, axis = 0)
        expanded_array.append(new)
        
        # extend param
        new = np.repeat(params_array[indices[0], :], repeat_times, axis = 0)
        expanded_params_array.append(new)
        
    # type
    print(' ----------- Types ---------- ')
    expanded_array = np.hstack(expanded_array)
    print('ori type:', types_array.shape)
    print('extend type:', expanded_array.shape)
    types_array = np.concatenate((types_array, expanded_array), axis = 0)
    print('total type:', types_array.shape)

    # param
    print(' ----------- Params ---------- ')
    expanded_params_array = np.vstack(expanded_params_array)
    print('ori param:', params_array.shape)
    print('extend param:', expanded_params_array.shape)
    params_array = np.concatenate((params_array, expanded_params_array), axis = 0)
    print('total param:', params_array.shape)
    return types_array, params_array


def main(json_file):
    
    # Load the configuration from the provided JSON file
    with open(json_file, 'r') as f:
        config = json.load(f)
    print(config)
    
    data_dir = config["paths"]["data_dir"]
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    ## Read in original dataset
    # Parameters
    filename = os.path.join(data_dir, config["original_dataset"]["params_filename"])
    params_array = np.load(filename).astype(np.float32)
    all_params = config["augmented_dataset"]["all_params"]
    sceening_params = config["augmented_dataset"]["selected_params"]
    selected_param_idx = [all_params.index(param) for param in sceening_params]
    params_array = params_array[:, selected_param_idx]

    # Pattern types
    pattern_types_array = np.load(os.path.join(data_dir,config["original_dataset"]["types_filename"]))
    pattern_types_array = pattern_types_array[:, 1]

    # Check size
    print(f"PDE Parameters: {params_array.shape}")
    print(f"Pattern types:  {pattern_types_array.shape}")

    ## Sort pattern_types_array in descending order and get the indices
    sorted_indices = np.argsort(pattern_types_array)[::-1]
    sorted_pattern_types_array = pattern_types_array[sorted_indices]
    sorted_params_array = params_array[sorted_indices]

    # Keep 3+ ring patterns
    filter_mask = sorted_pattern_types_array >= 5
    filtered_pattern_types_array = sorted_pattern_types_array[filter_mask]
    filtered_params_array = sorted_params_array[filter_mask]

    # Repeat the chosen parameters
    total_num = config["augmented_dataset"]["num_to_augment"]
    filtered_pattern_types_array = np.array(filtered_pattern_types_array)
    expanded_pattern_types_array, expanded_params_array = expand_to_num(filtered_pattern_types_array, filtered_params_array, total_num)

    # Save
    filename = output_dir + 'augment_params.npy'
    np.save(filename, np.array(expanded_params_array))
    filename = output_dir + 'augment_types.npy'
    np.save(filename, np.array(expanded_pattern_types_array))
    filename = os.path.join(output_dir, 'augment_params.txt')
    np.savetxt(filename, expanded_params_array, delimiter=',', fmt='%0.8f')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment the original dataset, parameters forming 3 or more rings will be preturbed, followed by numerical simulations')
    parser.add_argument('config', type=str, help='JSON configuration filename')
    args = parser.parse_args()
    main(args.config)