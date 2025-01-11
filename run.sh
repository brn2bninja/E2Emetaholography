#!/bin/bash

#SBATCH -p gpu
#SBATCH -J e2e_2025
#SBATCH -o slurm_logs/single_%j.out
#SBATCH -t 50:00:00
#SBATCH --nodelist=node35
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50gb
#SBATCH --cpus-per-gpu=12

CUDA_VISIBLE_DEVICES=0 python operate_single.py