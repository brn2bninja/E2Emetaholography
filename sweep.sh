#!/bin/bash

#SBATCH -p gpu
#SBATCH -J e2e_sweep_lr
#SBATCH -o slurm_logs/sweep_%j.out
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node35
#SBATCH --mem=12gb
#SBATCH --cpus-per-gpu=12

python operate_sweep.py