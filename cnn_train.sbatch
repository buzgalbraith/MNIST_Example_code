#!/bin/bash 

#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=cnn_training
#SBATCH --output=cnn_training.out

module purge ## purge modules that we are not using 
module load python/intel/3.8.6 ## load python module
python ./cnn_train.py ## run python training script.
echo "finished training and saving model"
