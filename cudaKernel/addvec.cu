
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

double cpuTimer() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void initialData(float *arr, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; ++i) {
        arr[i] = (float)(rand()) / RAND_MAX;
    }
}

void checkResult(float *host, float *gpu, const int N) {
    const double epsilon = 1.0E-8;
    bool match = 1;

    for(int i = 0; i < N; ++i) {
        if (abs(host[i] - gpu[i]) > epsilon) {
            match = 0;
            printf("Vector do not match!\n");
            printf("host %5.2f, gpu %5.2f at current %d\n", host[i], gpu[i], i);
            break;
        }
    }

    if(match)
        printf("Vector match.\n");
}

void AddVecOnHost(float *A, float *B, float *C, const int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void AddVecOnGPU(float *A, float *B, float *C, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv) {
    int nSize = 128 * 1024 * 1024;
    printf("Vector size : %d\n", nSize);

    // == on host ==
    size_t nBytes = nSize * sizeof(float);
    float *h_A, *h_B, *hostResult, *gpuResult;
    double iStart, ElapsedTime;

    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostResult = (float *)malloc(nBytes);
    gpuResult = (float *)malloc(nBytes);

    initialData(h_A, nSize);
    initialData(h_B, nSize);

    memset(hostResult, 0, nBytes);
    memset(gpuResult, 0, nBytes);

    iStart = cpuTimer();

    AddVecOnHost(h_A, h_B, hostResult, nSize);

    ElapsedTime = cpuTimer() - iStart;

    printf("Elapsed Time in AddVecOnHost: %f\n", ElapsedTime);
    // == ==== ==

    // == on device ==
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    //data transfer: Host -> device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((nSize + block.x - 1) / block.x);

    iStart = cpuTimer();

    AddVecOnGPU<<<grid, block>>>(d_A, d_B, d_C, nSize);
    cudaDeviceSynchronize();
    
    ElapsedTime = cpuTimer() - iStart;
    
    printf("Elapsed Time in AddVecOnGPU<<<%d, %d>>> : %f\n", grid.x, block.x, ElapsedTime);
    cudaMemcpy(gpuResult, d_C, nBytes, cudaMemcpyDeviceToHost);
	// == ==== ==

    checkResult(hostResult, gpuResult, nSize);

    free(h_A); free(h_B); free(hostResult); free(gpuResult);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}

