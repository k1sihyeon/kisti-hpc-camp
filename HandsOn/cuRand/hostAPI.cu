/*
 * This program uses the host CURAND API to generate 100 
 * pseudorandom floats.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
int main(void)
{
	size_t n = 100;
	size_t i;
	curandGenerator_t gen;
	float *devData, *hostData;
	
	/* Allocate n floats on host */
	hostData=(float*)malloc(n*sizeof(float));
	cudaMalloc((float**)&devData,n*sizeof(float));

	/* Create pseudo-random number generator */
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	/* Set seed */
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	/* Generate n floats on device */
	curandGenerateUniform(gen, devData, n);
	
	/* Copy device memory to host */
	cudaMemcpy(hostData, devData, n*sizeof(float), cudaMemcpyDefault);

	/* Show result */
	for(i=0;i<n;++i)
		printf("%1.4f ", hostData[i]);
	printf("\n");

	/* Cleanup */
	curandDestroyGenerator(gen);
	cudaFree(devData);
	free(hostData);

	return 0;
}
