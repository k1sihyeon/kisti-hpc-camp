#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/sequence.h>
int main(void)
{
	size_t N=10;
	// create a device_ptr
	thrust::device_ptr<int> dev_ptr=thrust::device_malloc<int>(N);
	thrust::sequence(dev_ptr, dev_ptr+N);
	for(int i=0;i<N;++i)
		std::cout<<dev_ptr[i]<<" ";
	std::cout<<std::endl;

	// extract raw pointer from deviced_ptr
	int *raw_ptr=thrust::raw_pointer_cast(dev_ptr);

	return 0;
}
