#!/bin/sh
#SBATCH -J gpu_serial
#SBATCH --time=00:05:00
#SBATCH -p edu
#SBATCH -o gpu_serial_%j.o
#SBATCH -e gpu_serial_%j.e
#SBATCH --gres=gpu:1
#SBATCH --comment=etc
./HelloCUDA
