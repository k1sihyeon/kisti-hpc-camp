#!/bin/sh
#SBATCH -J addvecu
#SBATCH --time=00:05:00
#SBATCH --nodes 1
#SBATCH --tasks-per-node=8
#SBATCH -p mig_amd_a100_4
#SBATCH -o %x_%j.o
#SBATCH -e %x_%j.e
#SBATCH --gres=gpu:2
#SBATCH --comment=etc

export OMP_NUM_THREADS=8
./addvec
exit 0
