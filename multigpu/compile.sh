nvcc -c Print.cu
#mpiCC -c traditionalMPI.c -I/apps/cuda/11.2/include
#mpiCC traditionalMPI.o Print.o -o traditionalMPI.x -L/apps/cuda/11.2/lib64 -lcudart


mpiCC -c CUDAawareMPI.c -I/apps/cuda/11.4/include
mpiCC CUDAawareMPI.o Print.o -o CUDAawareMPI.x -L/apps/cuda/11.4/lib64 -lcudart



