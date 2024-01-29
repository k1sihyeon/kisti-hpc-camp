#!/bin/bash
#SBATCH -J MPI_job
###SBATCH -p cas_v100nv_8
#SBATCH -p edu
#SBATCH -o %x_%j.o
#SBATCH -e %x_%j.e
#SBATCH --time 00:05:00
#SBATCH -n 4
#SBATCH --tasks-per-node=4
###SBATCH --gres=gpu:2
#SBATCH --comment etc

srun ./HelloMPI.x
