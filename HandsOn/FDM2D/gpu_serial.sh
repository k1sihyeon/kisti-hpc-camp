#!/bin/sh
#SBATCH -J gpu_serial
#SBATCH --time=00:05:00
#SBATCH -p cas_v100_4
#SBATCH -N 1
#SBATCH -n 40
#SBATCH -o gpu_serial_%j.out
#SBATCH -e gpu_serial_%j.err
#SBATCH --gres=gpu:2
#SBATCH --comment=etc

./$1
#$1
#nvprof ./pinned.x
