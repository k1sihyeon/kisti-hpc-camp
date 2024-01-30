#include <stdio.h>

__global__ void GPUKernel(int arg) {
	
	printf("Input Value (on GPU) = %d \n", arg);
}

int main(void) {
	printf("Call Kernel Function! \n");
	
	GPUKernel<<<1, 1>>>(1);
	GPUKernel<<<1, 1>>>(2);
	cudaDeviceSynchronize();

	return 0;
}
