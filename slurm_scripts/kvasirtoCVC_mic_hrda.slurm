#!/bin/bash
#SBATCH -p grantgpu -A g2024a219g
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100|gpuv100"
#SBATCH -o jobs/kvasirtoCVC_mic_hrda.out
hostname
source deactivate
module load python/python-3.8.18
source ~/venv/hrda/bin/activate
python --version
echo 'START'
python run_experiments.py --config configs/mic/kvasirtoCVC_mic_hrda.py
echo 'END'
