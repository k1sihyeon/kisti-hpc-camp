#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <iostream>

int main(void)
{
	size_t N=10;
	// row pointer to device memory
	int * raw_ptr;
	cudaMalloc((void**)&raw_ptr, N*sizeof(int));

	// warp raw pointer with a device_ptr
	thrust::device_ptr<int> dev_ptr(raw_ptr);

	// use device_ptr in thrust algorithms
	thrust::fill(dev_ptr, dev_ptr+N, (int) 0);
	for(int i=0;i<N;++i)
		std::cout<<"dev_ptr["<<i<<"]="<<dev_ptr[i]<<std::endl;
	return 0;
}
