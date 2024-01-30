#include <cuda_runtime.h>
#include <stdio.h>
int main(void)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Device : \"%s\"\n", deviceProp.name);

	int driverVersion=0, runtimeVersion=0;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("\tCUDA Driver Version / Runtime Version %d.%d / %d.%d\n",
			driverVersion/1000, (driverVersion%100)/10,
			runtimeVersion/1000,(runtimeVersion%100)/10);

	// compute capability major/minor number
	printf("\tCUDA Capability Major/Minor version number: %d.%d\n",
				deviceProp.major, deviceProp.minor);
	// Global memory(in bytes)
	printf("\tTotal amount of global memory: %.2f GBytes (%llu bytes)\n",
			(float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
			(unsigned long long)deviceProp.totalGlobalMem);
	// Clock Rate(kHz)
	printf("\tGPU Clock rate : \t%.0f MHz(%0.2f GHz)\n",
			deviceProp.clockRate*1e-3f, deviceProp.clockRate*1e-6);
	// memory frequency ( in KHz)
	printf("\tMemory Clock rate : \t%.0f MHz\n",
			deviceProp.memoryClockRate*1e-3);
	// memory bus interface(in bits)
	printf("\tMemory Bus Width : \t%d-bit\n", deviceProp.memoryBusWidth);

	//L2 cache size(in bytes)
	if(deviceProp.l2CacheSize)
		printf("\tL2 Cache Size:\t%d bytes\n",deviceProp.l2CacheSize);

	// Constant Memory(in bytes)
	printf("\tTotal amount of constant memory:\t%lu bytes\n",deviceProp.totalConstMem);

	// Shared memory size & # of SM
	printf("\tTotal amount of shared memory per block:\t%lu bytes\n",
		deviceProp.sharedMemPerBlock);
	printf("\tTotal amound of shared memory per SM:\t%lu bytes\n",
		deviceProp.sharedMemPerMultiprocessor);
	printf("\tNumber of SMs:\t%lu \n", deviceProp.multiProcessorCount);

	// # of register per block
	printf("\tTotal number of registers available per block:\t%d\n",
		deviceProp.regsPerBlock);
	printf("\tWarp Size:\t%d\n",deviceProp.warpSize);
	
	// Max # of threads per SM
	printf("\tMaximum number of threads per multiprocessor:\t%d\n",
		deviceProp.maxThreadsPerMultiProcessor);

	// Max # of threads per block
	printf("\tMaximum number of threads per block:\t%d\n",
		deviceProp.maxThreadsPerBlock);

	// Max # of threads of each dimension per block
	printf("\tMaximum sizes of each dimension of a block:\t%d x %d x %d\n",
		deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

	// Max # of threads of each dimension in Grid
	printf("\tMaximum sizes of each dimension of a grid:\t%d x %d x %d\n",
		deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	
	return 0;		
}
