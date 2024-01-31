#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define N (10)

int main(void){
	float h_x[N], h_y[N];
	float *d_x, *d_y;
	cudaMalloc((float**)&d_x, N*sizeof(float));
	cudaMalloc((float**)&d_y, N*sizeof(float));
	for(int i=0;i<N;i++){
		h_x[i]=10.0f,	h_y[i]=1.0f;
	}
//	cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyDefault);
//	cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyDefault);

	cublasSetVector(N, sizeof(float), h_x, 1, d_x,1);
	cublasSetVector(N, sizeof(float), h_y, 1, d_y,1);

	cublasStatus_t stat;
	cublasHandle_t handle;

	cublasCreate(&handle);
	const float alpha=2.0;
	stat = cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("Error =======\n");
	    cublasDestroy(handle);
    	cudaFree(d_x), cudaFree(d_y);
		return EXIT_FAILURE;
	}
	cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDefault);

	for(int i=0;i<N;i++)
		printf("%.2f ",h_y[i]);
	printf("\n");

	cublasDestroy(handle);
	cudaFree(d_x), cudaFree(d_y);	
	return EXIT_SUCCESS;	
}

