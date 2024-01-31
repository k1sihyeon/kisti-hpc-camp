#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>
#include <cuda_runtime.h>
int main(void)
{
	cudaError_t cudaStat=cudaSuccess;
	const int n=5, nnz=13;
	int *d_cooRowInd, *d_cooColInd;
	float *d_A;
	int cooRowInd[nnz]={0,0,0,0,0,1,1,2,2,3,3,4,4};
	int cooColInd[nnz]={0,1,2,3,4, 0,1, 0,2, 0,3, 0,4 };
	float A[nnz]={4.0f, 1.0f, 2.0f, 0.5f, 2.0f, 1.0f, 0.5f, 2.0f, 3.0f, 0.5f, 0.625f, 2.0f, 16.0f};

	cudaStat=cudaMalloc((int**)&d_cooRowInd, nnz*sizeof(int));
	if(cudaStat != cudaSuccess) printf("bbb\n");
	cudaStat=cudaMalloc((int**)&d_cooColInd, nnz*sizeof(int));

	if(cudaStat != cudaSuccess){
		printf("aaaa\n");

	}
	
	cudaMalloc((float**)&d_A, nnz*sizeof(float));
	cudaMemcpy(d_cooRowInd, cooRowInd, nnz*sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(d_cooColInd, cooColInd, nnz*sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(d_A, A, nnz*sizeof(float), cudaMemcpyDefault);
	float alpah=1.0f, beta=1.0f;


// Vector initialize
	float x[n]={1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	float y[n]={1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	float hY[n];
	float *d_x, *d_y;
	cudaMalloc((float**)&d_x, sizeof(float)*n);
	cudaMalloc((float**)&d_y, sizeof(float)*n);
	cudaMemcpy(d_x, x, sizeof(float)*n, cudaMemcpyDefault);
	cudaMemcpy(d_y, y, sizeof(float)*n, cudaMemcpyDefault);

    cusparseHandle_t handle=NULL;
    cusparseStatus_t stat;
	size_t bufferSize=0;
	cusparseCreate(&handle);
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;

	cusparseCreateCoo(&matA, n,n,nnz,d_cooRowInd, d_cooColInd,d_A, CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F);

	cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_32F);
	cusparseCreateDnVec(&vecY, n, d_y, CUDA_R_32F);

	cusparseSpMV_bufferSize(	handle,
								CUSPARSE_OPERATION_NON_TRANSPOSE,
								&alpah,
								matA,
								vecX,
								&beta,
								vecY,
								CUDA_R_32F,
								CUSPARSE_MV_ALG_DEFAULT,
//								CUSPARSE_SPMV_COO_ALG1,
								&bufferSize);

	printf("bufferSize: %zd\n",bufferSize);
	void *dBuffer;
	cudaMalloc(&dBuffer, bufferSize);


	cusparseSpMV(	handle,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					(float*)&alpah,
					matA,
					vecX,
					(float*)&beta,
					vecY,
					CUDA_R_32F,
					CUSPARSE_MV_ALG_DEFAULT,
					dBuffer
				);

	cudaMemcpy(hY,d_y,sizeof(float)*n, cudaMemcpyDefault);
	for(int i=0;i<n;i++)
		printf("%f ",(float)hY[i]);
	printf("\n");

	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecY);
	cusparseDestroySpMat(matA);
	cusparseDestroy(handle);

	return EXIT_SUCCESS;
}
