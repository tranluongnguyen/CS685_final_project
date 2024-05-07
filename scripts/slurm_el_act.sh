#!/bin/bash
#SBATCH --job-name=gp
#SBATCH --output=/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/tdngo_ws/gpt_ws/slurm_log/%A_out.out         # Output file
#SBATCH --error=/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/tdngo_ws/gpt_ws/slurm_log/%A_err.out         # Output file
#SBATCH --partition=dcs-2024
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=32           # Number of CPUs per task
#SBATCH --mem=128G                    # Memory per node
#SBATCH --time=06:00:00             # Time limit
#SBATCH --mail-type=all
#SBATCH --mail-user=ductuan.ngo99@gmail.com

# Load any necessary modules
cd /gpfs/u/home/LMCG/LMCGljnn/scratch-shared/tdngo_ws/gpt_ws/CS685_final_project
source /gpfs/u/home/LMCG/LMCGljnn/scratch/miniconda3-ppc64le/etc/profile.d/conda.sh
conda activate LucidDreamer

# Your commands here
# torchrun --standalone --nproc_per_node=4 train.py
torchrun --standalone --nproc_per_node=4 active_forgetting_train.py config/af_train_gpt2.py

# End of script

# Your commands here

# torchrun --standalone --nproc_per_node=2 active_forgetting_train.py config/af_train_gpt2.py