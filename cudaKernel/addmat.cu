#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
//#include <stdbool.h>
double cpuTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialData(float *arr, int size)
{
    time_t t;
    srand((unsigned)time(&t));  // seed
    for(int i=0;i<size;++i)
        arr[i]=(float)(rand())/RAND_MAX;
}

void AddMatOnHost_1D(float *A, float *B, float *C, const int nx, const int ny)
{
#pragma omp parallel for
	for(int i=0;i<ny;i++){
		for(int j=0;j<nx;j++){
			C[j] = A[j] + B[j];
		}
		A += nx, 	B += nx,	C += nx;
	}
}
void AddMatOnHost(float *A, float *B, float *C, const int M, const int N)
{
	float (*pA)[N]=(float (*)[N])A;
	float (*pB)[N]=(float (*)[N])B;
	float (*pC)[N]=(float (*)[N])C;
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			pC[i][j]=pA[i][j]+pB[i][j];
		}
	}
}

__global__ void AddMatOnGPU(float *A, float *B, float *C, const int nx, const int ny)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = col+ row*nx;
	if(col<nx && row<ny) C[idx] = A[idx] + B[idx];
}
template <int col> __global__ void AddMatOnGPU(float *A, float *B, float *C, int M, int N)
{
	int idx_x=blockIdx.x*blockDim.x + threadIdx.x;
	int idx_y=blockIdx.y*blockDim.y + threadIdx.y;
	float (*pA)[col]=(float (*)[col])A;
	float (*pB)[col]=(float (*)[col])B;
	float (*pC)[col]=(float (*)[col])C;
	if(idx_x<N && idx_y<M) pC[idx_y][idx_x] = pA[idx_y][idx_x]+pB[idx_y][idx_x];
}
void checkResult(float *host, float *gpu, const int M, const int N)
{
	const double epsilon = 1.0e-8;
	bool match=1;
	float (*pA)[N]=(float (*)[N])host;
	float (*pB)[N]=(float (*)[N])gpu;
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			if(abs(pA[i][j] - pB[i][j])>epsilon)
			{
				match=0;
				printf("Matrix do not match\n");
				printf("host %5.2f, gpu %5.2f at current [%d][%d]\n", pA[i][j], pB[i][j], i, j);
				return;
			}
		}
	}
	if(match) printf("Matrices match.\n");
}

int main(int argc, char **argv)
{
	const int M=16*1024, N=16*1024;
	printf("Matrix size : %d\n",M*N);
	size_t nBytes=sizeof(float)*(M*N);
	/***** on Host **********/
	float h_A[M*N], h_B[M*N], hostResult[M*N], gpuResult[M*N];
	double iStart, ElapsedTime;
	initialData(h_A, M*N);
	initialData(h_B, M*N);

	iStart=cpuTimer();
	AddMatOnHost(h_A, h_B, hostResult, M,N);
//	AddMatOnHost_1D(h_A, h_B, hostResult, M, N);
	ElapsedTime=cpuTimer()-iStart;
	printf("Elapsed Time(CPU) : %5.2f sec.\n",ElapsedTime);

	/******* on GPU ********/
	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_B, nBytes);
	cudaMalloc((float**)&d_C, nBytes);
	// Data transfer : Host -> Device
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyDefault);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyDefault);

	int dimx=32, dimy=32;
	if(argc>2){
		dimx=atoi(argv[1]), dimy=atoi(argv[2]);
	}
	dim3 block(dimx,dimy);
	dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
	cudaEvent_t start, stop;
	cudaEventCreate(&start), cudaEventCreate(&stop);
	float Etime;
	cudaEventRecord(start);
	AddMatOnGPU<N><<<grid,block>>>(d_A, d_B, d_C, M, N);
//	AddMatOnGPU<<<grid,block>>>(d_A, d_B, d_C, M, N);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&Etime, start, stop);
	cudaEventDestroy(start), cudaEventDestroy(stop);
	printf("Elapsed Time(GPU) : %5.2f ms\n",Etime);
	cudaMemcpy(gpuResult, d_C, nBytes, cudaMemcpyDefault);
	checkResult(hostResult, gpuResult,M,N);
	cudaFree(d_A), cudaFree(d_B), cudaFree(d_C);
	return 0;
	
}
