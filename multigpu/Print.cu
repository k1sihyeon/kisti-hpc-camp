#include <stdio.h>
#include <cuda_runtime.h>
__global__ void Print(int *buf, int N)
{
    for(int i=0;i<N;i++)
        printf("%d ",buf[i]);
    printf("\n");
}

void kernel_wrapper(int *buf, int N)
{
	Print<<<1,1>>>(buf, N);
	cudaDeviceSynchronize();
}


