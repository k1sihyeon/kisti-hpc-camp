#!/bin/bash
#SBATCH -J serial_job
###SBATCH -p cas_v100nv_8
#SBATCH -p edu
#SBATCH -o %x_%j.o
#SBATCH -e %x_%j.e
#SBATCH --time 00:05:00
###SBATCH --gres=gpu:2
#SBATCH --comment etc

/scratch/kedu14/job_sample/serial/HelloOnCPU.x
