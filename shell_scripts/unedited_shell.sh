#!/bin/bash
#SBATCH --job-name=lebwohllasher
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --account=PHYS030544
#SBATCH --mem-per-cpu=100M

python lebwohlLasher.py 100 10 0.5 0
