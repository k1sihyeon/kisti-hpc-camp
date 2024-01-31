#!/bin/bash
#SBATCH -J mpi_gpu_job
#SBATCH -p mig_amd_a100_4
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time 00:05:00
#SBATCH --gres=gpu:1
#SBATCH --comment etc     # See Application SBATCH options name table's

#module purge
#module load gcc/8.3.0 cuda/11.2 cudampi/openmpi-3.1.5

export OMPI_MCA_pml=^ucx
export OMPI_MCA_osc=rdma

ulimit -s unlimited

#srun ./traditionalMPI.x
srun ./CUDAawareMPI.x

