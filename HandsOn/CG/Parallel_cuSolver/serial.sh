#!/bin/sh
#SBATCH -J gpu_serial
#SBATCH --time=00:10:00
####SBATCH -p ivy_v100_2
#SBATCH -p cas_v100nv_4
#SBATCH -o gpu_serial_%j.out
#SBATCH -e gpu_serial_%j.err
#SBATCH --gres=gpu:1
#SBATCH --comment=etc
module purge
module add gcc/8.3.0 cuda/10.1
ulimit -s unlimited
./cg

