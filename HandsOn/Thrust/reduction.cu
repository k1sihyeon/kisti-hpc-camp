#include <thrust/reduce.h>
#include <thrust/device_vector.h>
int main(void)
{
	int arr[]={1,2,3,4,5};
	int n=sizeof(arr)/sizeof(int);
	thrust::device_vector<int> dest(arr,arr+n);
	for(int i=0;i<dest.size();++i)
		std::cout<<dest[i]<<" ";
	std::cout<<std::endl;

	int sum=thrust::reduce(dest.begin(), dest.end(),(int)0, thrust::plus<int>());
	std::cout<<"Sum="<<sum<<std::endl;
	return 0;
}
