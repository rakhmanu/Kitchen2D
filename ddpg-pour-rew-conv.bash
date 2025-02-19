#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=64g
#SBATCH --partition=rleap_gpu_24gb
#SBATCH --output=/work/rleap1/ulzhalgas.rakhman/%A.txt

python3 ddpg-pour-rew-conv.py