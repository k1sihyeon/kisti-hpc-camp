#!/bin/sh
#SBATCH -J OpenMp_job
#SBATCH --time=00:05:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o OpenMP_%j.o
#SBATCH -e OpenMp_%j.e
#SBATCH -p edu
#SBATCH --comment etc
export OMP_NUM_THREADS=4
./HelloOpenMP.x
