#include <stdio.h>

__device__ void PrintArray(int tid, int *A) {
	printf("A[%d] = %d\n", tid, A[tid]);
	A[tid] = A[tid] + 1;
	if (tid == 0)
		printf("=========================\n");
}

__global__ void Print(int *A) {

	int tid = threadIdx.x;
	PrintArray(tid, A);
}

int main() {
	int *d1_A, *d2_A, *h_A, *h_B;
	int size = 5; int i;
	
	h_A = (int*)malloc(size * sizeof(int));
	h_B = (int*)malloc(size * sizeof(int));
	
	for(i = 0; i < size; i++)
		h_A[i] = i;

	//device mem alloc
	cudaSetDevice(0);
	cudaMalloc((int **)&d1_A, size * sizeof(int));

	cudaSetDevice(1);
	cudaMalloc((int **)&d2_A, size * sizeof(int));
	//==

	//data transfer host -> dev0
	cudaSetDevice(0);
	cudaMemcpy(d1_A, h_A, size * sizeof(int), cudaMemcpyHostToDevice); //cudaMemcpyDefault
	Print<<<1, 5>>>(d1_A);
	cudaDeviceSynchronize();

	//data transfer dev0 -> 1
	cudaMemcpy(d2_A, d1_A, size * sizeof(int), cudaMemcpyDeviceToDevice); //
	cudaSetDevice(1);
	Print<<<1, 5>>>(d2_A);
	cudaDeviceSynchronize();

	//data transfer dev1 -> host
	cudaMemcpy(h_B, d2_A, size * sizeof(int), cudaMemcpyDeviceToHost);
	for(i = 0; i < size; i++)
		printf("h_B[%d] = %d\n", i, h_B[i]);
	
	cudaFree(d1_A);
	cudaFree(d2_A);
	
	cudaDeviceReset();
	
	return 0;
}
