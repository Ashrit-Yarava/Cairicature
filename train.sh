#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=cairicature
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=18000
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ary24/Painter/output.%N.%j.out

cd /scratch/ary24/Painter/
source ~/.bashrc
conda activate ai
python3 train.py