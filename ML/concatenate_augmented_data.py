import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse

def main(json_file):
    
    # Load the configuration from the provided JSON file
    with open(json_file, 'r') as f:
        config = json.load(f)
    print(config)
    
    all_params = config["dataset"]["all_params"]
    sceening_params = config["dataset"]["selected_params"]
    
    # -------------- Dataset 1 (original) --------------
    data_directory_1 = config["paths"]["data_dir_1"]

    # RFP profiles
    data_array = np.load(os.path.join(data_directory_1, config["paths"]["output_filename_1"]))
    data_array = data_array.reshape([-1, 3, 201])
    norm_data_array = data_array / data_array.max(axis=2, keepdims=True)

    # Parameters 
    params_array = np.load(os.path.join(data_directory_1, config["paths"]["params_filename_1"]))
    selected_param_idx = [all_params.index(param) for param in sceening_params]
    params_array = params_array[:, selected_param_idx]

    # Pattern types
    pattern_types_array = np.load(os.path.join(data_directory_1, config["paths"]["types_filename_1"]))

    print(' ------ Original ------', end='\')
    print(f"RFP profiles: {norm_data_array.shape}")
    print(f"Parameters: {params_array.shape}")
    print(f"Pattern types:  {pattern_types_array.shape}")

    # -------------- Dataset 2 (augmented) --------------
    # Read in augmented data
    data_directory_2 = config["paths"]["data_dir_2"]

    # RFP profiles
    data_array_2 = np.load(os.path.join(data_directory_2, config["paths"]["output_filename_2"]))
    data_array_2 = data_array_2.reshape([-1, 9, 201])
    data_array_2 = data_array_2[:, 0:3, :]
    norm_data_array_2 = data_array_2 / data_array_2.max(axis=2, keepdims=True)

    # Parameters 
    params_array_2 = np.load(os.path.join(data_directory_2, config["paths"]["params_filename_2"]))
    params_array_2 = params_array_2[:, selected_param_idx]

    # Pattern types
    pattern_types_array_2 = np.load(os.path.join(data_directory_2, config["paths"]["types_filename_2"]))

    print(' ------ Augmented ------', end='\n')
    print(f"RFP profiles: {norm_data_array_2.shape}")
    print(f"Parameters: {params_array_2.shape}")
    print(f"Pattern types:  {pattern_types_array_2.shape}")


    # -------------- Dataset 3 (augmented) --------------
    data_directory_3 = config["paths"]["data_dir_3"]

    # RFP profiles
    data_array_3 = np.load(os.path.join(data_directory_3, config["paths"]["output_filename_3"]))
    data_array_3 = data_array_3.reshape([-1, 9, 201])
    data_array_3 = data_array_3[:, 0:3, :]
    norm_data_array_3 = data_array_3 / data_array_3.max(axis=2, keepdims=True)

    # Parameters 
    params_array_3 = np.load(os.path.join(data_directory_3, config["paths"]["params_filename_3"]))
    params_array_3 = params_array_3[:, selected_param_idx]

    # Pattern types
    pattern_types_array_3 = np.load(os.path.join(data_directory_3, config["paths"]["types_filename_3"]))

    print(' ------ Augmented ------ ', end='\n')
    print(f"RFP profiles: {norm_data_array_3.shape}")
    print(f"Parameters: {params_array_3.shape}")
    print(f"Pattern types:  {pattern_types_array_3.shape}")


    # Concatenate
    output_dir = config["paths"]["output_dir"]

    # Outputs
    data_array_cont = np.concatenate((data_array, data_array_2, data_array_3), axis=0)
    filename = os.path.join(output_dir, 'all_outputs_cont.npy')
    np.save(filename, np.array(data_array_cont))

    # Params
    params_array_cont = np.concatenate((params_array, params_array_2, params_array_3), axis=0)
    filename = os.path.join(output_dir, 'all_params_cont.npy')
    np.save(filename, np.array(params_array_cont))
    filename = os.path.join(output_dir, 'all_params_cont.txt')
    np.savetxt(filename, params_array_cont, delimiter=',', fmt='%0.8f')

    # Types
    pattern_types_array_cont = np.concatenate((pattern_types_array, pattern_types_array_2, pattern_types_array_3), axis=0)
    filename = os.path.join(output_dir, 'all_types_cont.npy')
    np.save(filename, np.array(pattern_types_array_cont))

    # Plot dataset histrogram
    counts, bin_edges, _ = plt.hist(pattern_types_array_cont[:, 1])
    figname = os.path.join(output_dir, 'hist.png')
    plt.savefig(figname)
          
    num_augmented = len(pattern_types_array_cont) - len(params_array)
    print('Augmented data #: ', num_augmented)
    print('Final dataset size: ', len(pattern_types_array_cont))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate the original and augments datasets')
    parser.add_argument('config', type=str, help='JSON configuration filename')
    args = parser.parse_args()
    main(args.config)

