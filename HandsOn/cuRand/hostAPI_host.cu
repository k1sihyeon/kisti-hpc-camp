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
	float *hostData;
	
	/* Allocate n floats on host */
	hostData=(float*)malloc(n*sizeof(float));

	/* Create pseudo-random number generator */
	curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	/* Set seed */
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	/* Generate n floats on device */
	curandGenerateUniform(gen, hostData, n);
	

	/* Show result */
	for(i=0;i<n;++i)
		printf("%1.4f ", hostData[i]);
	printf("\n");

	/* Cleanup */
	curandDestroyGenerator(gen);
	free(hostData);

	return 0;
}
