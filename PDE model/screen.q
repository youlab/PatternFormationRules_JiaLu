#!/bin/bash
#SBATCH -o slurm.out
#SBATCH -e slurm.err
#SBATCH -a 1-300
#SBATCH -c 1
#SBATCH --partition=scavenger
#SBATCH --mem=2GB 
/opt/apps/rhel8/matlabR2021a/bin/matlab -nodesktop -nodisplay -singleCompThread -r "rank=$SLURM_ARRAY_TASK_ID;screening_main_steps_radial;exit"
