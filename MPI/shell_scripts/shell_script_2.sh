#!/bin/bash
#SBATCH --job-name=lebwohllasher
#SBATCH --partition=teach_cpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --account=PHYS030544
#SBATCH --mem-per-cpu=100M

mpirun -np 3 python c1e_c2o_mpi.py 100 10 0.5 0

