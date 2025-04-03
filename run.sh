#!/bin/bash

#SBATCH -p gpu
#SBATCH -J e2e_2025
#SBATCH -o slurm_logs/single_%j.out
#SBATCH -t 50:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node31
#SBATCH --mem=12gb
#SBATCH --cpus-per-gpu=12

python operate_single.py