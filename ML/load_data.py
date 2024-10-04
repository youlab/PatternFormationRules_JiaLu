import os
import numpy as np
from tqdm import tqdm
import scipy.signal as signal
from scipy.io import loadmat
from joblib import Parallel, delayed

# Load and process data from .mat files
def load_and_process_data(folder_name, file, seeding_v):
    
    # Grab file
    file_path = os.path.join(folder_name, file)
    # Return None for both outputs and params if the file is empty
    if os.path.getsize(file_path) == 0:
        return None, None  
    
    # Load .mat file
    mat_contents = loadmat(file_path)
    param = mat_contents['param']

    # Preprocess the profile
    length = 1001
    skip = 5 
    
    # Get the last line of hist_Ce, hist_Ly, and hist_T
    last_line_Ce = param['hist_Ce'][0, 0][0:length:skip, -1]
    total_RFP = param['hist_RFP'][0, 0][0:length:skip, -1]   
    total_CFP = param['hist_CFP'][0, 0][0:length:skip, -1]
    
    # Filter off negative values
    last_line_Ce[last_line_Ce < 0] = 0
    total_RFP[total_RFP < 0] = 0
    total_CFP[total_CFP < 0] = 0
    
    # Set NaN values to 0
    last_line_Ce[np.isnan(last_line_Ce)] = 0
    total_RFP[np.isnan(total_RFP)] = 0
    total_CFP[np.isnan(total_CFP)] = 0
    
    # Construct full 1D profiles
    flipped_Ce = np.flip(last_line_Ce)
    flipped_RFP = np.flip(total_RFP)
    flipped_CFP = np.flip(total_CFP)
    
    concat_Ce = np.concatenate((flipped_Ce, last_line_Ce))
    concat_RFP = np.concatenate((flipped_RFP, total_RFP))
    concat_CFP = np.concatenate((flipped_CFP, total_CFP))

    # Normalize               
    data_C_norm = (concat_Ce - np.min(concat_Ce)) / (np.max(concat_Ce) - np.min(concat_Ce))
    data_RFP_norm = (concat_RFP - np.min(concat_RFP)) / (np.max(concat_RFP) - np.min(concat_RFP))
    data_CFP_norm = (concat_CFP - np.min(concat_CFP)) / (np.max(concat_CFP) - np.min(concat_CFP))

    # Get pattern type of the originals
    peaks_Ce, _ = signal.find_peaks(data_C_norm, distance=5, width = 3, height = 0.1*np.max(data_C_norm), prominence=0.03*np.max(data_C_norm))
    peaks_RFP, _ = signal.find_peaks(data_RFP_norm, distance=5, width = 3, height = 0.03*np.max(data_RFP_norm), prominence=0.03*np.max(data_RFP_norm))
    peaks_CFP, _ = signal.find_peaks(data_CFP_norm, distance=5, width = 3, height = 0.03*np.max(data_CFP_norm), prominence=0.03*np.max(data_CFP_norm))

    params = np.column_stack([
        param['DC'][0,0][0][0], param['DN'][0,0][0][0], param['DA'][0,0][0][0], param['DB'][0,0][0][0],
        param['aC'][0,0][0][0], param['aA'][0,0][0][0], param['aB'][0,0][0][0], param['aT'][0,0][0][0],
        param['aL'][0,0][0][0], param['bN'][0,0][0][0], param['dA'][0,0][0][0], param['dB'][0,0][0][0],
        param['dT'][0,0][0][0], param['dL'][0,0][0][0], param['k1'][0,0][0][0], param['k2'][0,0][0][0],
        param['KN'][0,0][0][0], param['KP'][0,0][0][0], param['KT'][0,0][0][0], param['KA'][0,0][0][0],
        param['KB'][0,0][0][0], param['alpha'][0,0][0][0], param['beta'][0,0][0][0], param['Cmax'][0,0][0][0],
        param['a'][0,0][0][0], param['b'][0,0][0][0], param['m'][0,0][0][0], param['n'][0,0][0][0],
        param['Kphi'][0,0][0][0], param['l'][0,0][0][0], param['N0'][0,0][0][0], param['G1'][0,0][0][0],
        param['G2'][0,0][0][0], param['G3'][0,0][0][0], param['G4'][0,0][0][0], param['G5'][0,0][0][0],
        param['G6'][0,0][0][0], param['G7'][0,0][0][0], param['G8'][0,0][0][0], param['G9'][0,0][0][0],
        param['G10'][0,0][0][0], param['G11'][0,0][0][0], param['G12'][0,0][0][0], param['G13'][0,0][0][0],
        param['G14'][0,0][0][0], param['G15'][0,0][0][0], param['G16'][0,0][0][0], param['G17'][0,0][0][0],
        param['G18'][0,0][0][0], param['G19'][0,0][0][0], param['alpha_p'][0,0][0][0], param['beta_p'][0,0][0][0],
        seeding_v, #param['taskID'][0,0][0][0]  
    ])

    pattern_types = np.column_stack([len(peaks_Ce), len(peaks_RFP), len(peaks_CFP)])

    return [last_line_Ce, total_RFP, total_CFP], params, pattern_types

def process_file_chunk(folder_name, files_chunk, seeding_v):

    # Parallelize file chunks
    results = Parallel(n_jobs=-1)(delayed(load_and_process_data)(folder_name, file, seeding_v) for file in files_chunk)
    
    # Filter out results where outputs or params is None (e.g. empty files)
    results = [res for res in results if res[0] is not None and res[1] is not None]
    
    return results

# Process .mat files in each folder
def process_folder(folder_name, seeding_v):

    folder_path = os.path.join(root_path, folder_name)
    
    # Get all .mat files
    files = [file for file in os.listdir(folder_path) if file.endswith(".mat")]

    # Initialize variables
    temp_files = []
    CHUNK_SIZE = 100  # Define the size of each chunk

    # Process files in chunks
    for i in tqdm(range(0, len(files), CHUNK_SIZE)):

        # grab chuck of files
        files_chunk = [os.path.join(folder_path, file) for file in files[i:i + CHUNK_SIZE]]

        # Process each file chunk, skip unreadable files or files with no output
        chunk_results = process_file_chunk(folder_name, files_chunk, seeding_v)  

        # Filter out None results from chunk_results
        chunk_results = [result for result in chunk_results if result is not None and result[0] is not None]

        # If no valid results in the chunk, skip to the next chunk
        if not chunk_results:
            continue

        all_outputs = np.array([res[0] for res in chunk_results])
        all_params = np.array([res[1] for res in chunk_results])
        all_types = np.array([res[2] for res in chunk_results])
        print(all_outputs.shape)
        
        # Determine the maximum shape in each dimension
        max_output_shape = tuple(max(sizes) for sizes in zip(*[output.shape for output in all_outputs]))
        max_param_shape = tuple(max(sizes) for sizes in zip(*[param.shape for param in all_params]))
        max_type_shape = tuple(max(sizes) for sizes in zip(*[type_.shape for type_ in all_types]))

        # Pad the arrays to ensure consistent shapes
        all_outputs_padded = np.array([np.pad(output, [(0, m - s) for s, m in zip(output.shape, max_output_shape)], mode='constant') for output in all_outputs])
        all_params_padded = np.array([np.pad(param, [(0, m - s) for s, m in zip(param.shape, max_param_shape)], mode='constant') for param in all_params])
        all_types_padded = np.array([np.pad(type_, [(0, m - s) for s, m in zip(type_.shape, max_type_shape)], mode='constant') for type_ in all_types])

        # Save as temp files
        temp_filename = f"{folder_name}_temp_{i // CHUNK_SIZE + 1}.npz"
        temp_filename = os.path.join(folder_path, temp_filename)
        print(temp_filename)
        
        np.savez(temp_filename,
                    all_outputs=all_outputs_padded,
                    all_params=all_params_padded.squeeze(),
                    all_types=all_types_padded.squeeze())
        temp_files.append(temp_filename)

    return temp_files

# Put together all temp file data
def gether_all_data(folder_name):
    
    # Load all temporary files and concatenate the results
    all_outputs_combined = []
    all_params_combined = []
    all_types_combined = []
    
    # grab temp files
    temp_files = [temp_file for temp_file in os.listdir(folder_name) if 'temp' in temp_file]
    
    for temp_file in temp_files:
        temp_file_path = os.path.join(folder_name, temp_file)
        
        with np.load(temp_file_path, allow_pickle=True) as data:
            
            all_outputs = data['all_outputs']
            all_params = data['all_params']
            all_types = data['all_types']
            print(np.array(all_outputs).shape)
            
            if len(np.array(all_outputs).shape) == 3:                 
                all_outputs_combined.append(all_outputs)
                all_params_combined.append(all_params)
                all_types_combined.append(all_types)
         
    all_outputs_combined = np.vstack(all_outputs_combined)
    all_params_combined = np.vstack(all_params_combined)
    all_types_combined = np.vstack(all_types_combined)
    
    # Check final dataset size
    print(all_outputs_combined.shape)
    print(all_params_combined.shape)
    print(all_types_combined.shape)
    
    # Save the final concatenated results specific to this folder -- in each dataset's dir
    np.save(os.path.join(folder_name, f"all_outputs_{os.path.basename(folder_name)}.npy"), all_outputs_combined)
    np.save(os.path.join(folder_name, f'all_params_{os.path.basename(folder_name)}.npy'), all_params_combined)
    np.save(os.path.join(folder_name, f'all_types_{os.path.basename(folder_name)}.npy'), all_types_combined)

# Concatenate data from each folder, and save as one single dataset
def gather_final_dataset(folder_info, output_path):
    
    all_outputs_combined = []
    all_params_combined = []
    all_types_combined = []
    
    all_output_files = []
    all_params_files = []
    all_types_files = []
    
    for folder_path, _, _ in folder_info:
        print('Processing ... ', folder_path)
        for file in os.listdir(folder_path):

            if "all_outputs_" in file and file.endswith(".npy"):
                all_output_files.append(file)
                file_name = os.path.join(folder_path, file)
                data = np.load(file_name) 
                print(data.shape)
                all_outputs_combined.append(data)             

            if "all_params_" in file and file.endswith(".npy"):
                all_params_files.append(file)
                file_name = os.path.join(folder_path, file)
                data = np.load(file_name)
                print(data.shape)
                all_params_combined.append(data)

            if "all_types_" in file and file.endswith(".npy"):
                all_types_files.append(file)
                file_name = os.path.join(folder_path, file)
                data = np.load(file_name)
                print(data.shape)
                all_types_combined.append(data)
   
    all_outputs_combined = np.vstack(all_outputs_combined)
    all_params_combined = np.vstack(all_params_combined)
    all_types_combined = np.vstack(all_types_combined)
    
    print(all_outputs_combined.shape)
    print(all_params_combined.shape)
    print(all_types_combined.shape)

    print(output_path)
    
    # Concatenate the results, save in 1 file in output_path
    np.save(os.path.join(output_path, "all_outputs.npy"), all_outputs_combined)
    np.save(os.path.join(output_path, 'all_params.npy'), all_params_combined)
    np.save(os.path.join(output_path, 'all_types.npy'), all_types_combined)



def main():
    # Paths
    root_path = ".data/"
    output_path = root_path 
    os.makedirs(output_path, exist_ok=True)

    # Individual dataset  
    dataset_dir_1 = root_path 
    seeding_v_1 = 0.1 # seeding volume
    par_num = 53 # PDE parameter number

    # All folders to process
    folder_info = [(dataset_dir_1, seeding_v_1, par_num)]

    # Process each folder
    for folder_name, seeding_v, par_num in tqdm(folder_info): # for each folder
        print('Processing ....... ', folder_name)
        process_folder(folder_name, seeding_v) # process .mat files by batch and save as temp files
        gether_all_data(folder_name) # grab all temp files, concatenate

    # Concatenate data from each folder, and save as one single dataset
    gather_final_dataset(folder_info, output_path)

    # Check the sizes
    all_outputs = np.load(os.path.join(output_path, 'all_outputs.npy'))
    all_params  = np.load(os.path.join(output_path, 'all_params.npy'))
    all_types   = np.load(os.path.join(output_path, 'all_types.npy'))

    outputs_shape = all_outputs.shape
    params_shape  = all_params.shape
    types_shape   = all_types.shape

    print(f"All outputs: {outputs_shape}")
    print(f"All params: {params_shape}")
    print(f"All types:  {types_shape}")

    # Normalize profiles
    norm_outputs = all_outputs / all_outputs.max(axis=2, keepdims=True)
    filename = os.path.join(output_path, 'all_norm_outputs.npy')
    np.save(filename, norm_outputs)

    # Save params in txt format
    filename = os.path.join(output_path, 'all_params.txt')
    np.savetxt(filename, all_params, delimiter=',', fmt='%0.8f')
    print(filename)


main()