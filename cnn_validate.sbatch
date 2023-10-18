#!/bin/bash 

#SBATCH --nodes=1 # requests 3 compute servers
#SBATCH --gres=gpu:1 ## requesting 1 gpu
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=cnn_validation
#SBATCH --output=cnn_validation.out

module purge ## purge modules that we are not using 
module load python/intel/3.8.6 ## load python module
python ./cnn_validate.py ## run python training script.
echo "Job finished at: `date`" ## print the date and time the job finished

