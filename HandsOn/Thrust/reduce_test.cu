#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

template<int col> __global__ void SubMatOnGPU(float *A, float *B, const int M, const int N)
{
	int idx_x=blockIdx.x*blockDim.x + threadIdx.x;
	int idx_y=blockIdx.y*blockDim.y + threadIdx.y;
	float (*pA)[col]=(float (*)[col])A;
	float (*pB)[col]=(float (*)[col])B;
	if(idx_x<N && idx_y<M) pA[idx_y][idx_x]-= pB[idx_y][idx_x];
	__threadfence();
	float diff=thrust::reduce(thrust::device,(float*)pA,(float*)pA+M*N,0.,thrust::plus<float>());
	printf("diff : %lf \n",diff);
}
int main(void)
{
	const int M=4, N=4;
	float h_A[M*N], h_B[M*N];
	float *d_A, *d_B;
	cudaMalloc((float**)&d_A, M*N*sizeof(float));
	cudaMalloc((float**)&d_B, M*N*sizeof(float));
	h_A[0]=5.,	h_A[1]=6.,	h_A[2]=7.,	h_A[3]=8.;
	h_A[4]=5.,	h_A[5]=6.,	h_A[6]=7.,	h_A[7]=8.;
	h_A[8]=5.,	h_A[9]=6.,	h_A[10]=7.,	h_A[11]=8.;
	h_A[12]=5.,	h_A[13]=6.,	h_A[14]=7.,	h_A[15]=8.;

	h_B[0]=1.,	h_B[1]=2.,	h_B[2]=3.,	h_B[3]=4.;
	h_B[4]=1.,	h_B[5]=2.,	h_B[6]=3.,	h_B[7]=4.;
	h_B[8]=1.,	h_B[9]=2.,	h_B[10]=3.,	h_B[11]=4.;
	h_B[12]=1.,	h_B[13]=2.,	h_B[14]=3.,	h_B[15]=4.;
	dim3 block(2,2);
	dim3 grid(2,2);
	cudaMemcpy(d_A, h_A, M*N*sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(d_B, h_B, M*N*sizeof(float), cudaMemcpyDefault);
	SubMatOnGPU<N><<<grid,block>>>(d_A, d_B, M, N);
	cudaDeviceSynchronize();
	cudaFree(d_A),	cudaFree(d_B);
	return 0;
}
