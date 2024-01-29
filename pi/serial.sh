#!/bin/bash
#SBATCH -J serial_job
###SBATCH -p cas_v100nv_8
#SBATCH -p edu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time 01:00:00
###SBATCH --gres=gpu:2
#SBATCH --comment etc

/scratch/kedu14/job_sample/pi
