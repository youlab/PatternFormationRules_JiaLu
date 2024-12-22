# PatternFormationRules_JiaLu

This is the code respository for [**Decoding pattern formation rules by integrating mechanistic modeling and deep learning**](https://doi.org/10.1101/2024.09.02.610872?)
by Jia Lu, Nan Luo, Sizhe Liu, Kinshuk Sahu, Rohan Maddamsetti, Υasa Baig, Lingchong You

# **System Requirements**
The PDE simulation code is designed to run on CPUs, and the machine learning training code requires a GPU. To ensure optimal performance, we recommend running the PDE simulations in parallel on a compute cluster using SLURM. A sample SLURM script is provided in the PDE model folder. The allocated memory can be adjusted as needed; 2GB has been tested and found to be sufficient for most of the simulations. For loading and pre-processing simulation data, we suggest distributing the workload across multiple CPUs to improve efficiency. Additionally, we recommend using a GPU for machine learning inference to achieve fast processing times. 

# **Software Requirements**
Python, PyTorch (version 2.0 or higher), and MATLAB are required. Python 3.9, PyTorch 2.1, MATLAB R2022a, and CUDA version 12.2 have been tested to work. Refer to the setup.py file for a complete list of required Python packages. The installation should take minutes.

# **Simulation**
The MATLAB code generates a workspace file that contains all simulation parameters and the 1D profiles of all channels. These workspace files should be loaded and processed before being used to train a machine learning surrogate model, the code is included in the ML folder. See manuscript for detailed descriptions on simulation and inference times.
