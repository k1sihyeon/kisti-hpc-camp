#include <stdio.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <memory.h>
int main(void)
{

	cusparseHandle_t handle;
	int n=5, nnz=13;
	int *d_cooRowInd, *d_csrRow;
    int cooRowInd[nnz], csrRow[n+1];
	cudaMalloc((int**)&d_cooRowInd, sizeof(int)*nnz);
	cudaMalloc((int**)&d_csrRow, sizeof(int)*(n+1));

	cooRowInd[0]=0,	cooRowInd[1]=0,	cooRowInd[2]=0,	cooRowInd[3]=0,	cooRowInd[4]=0;
	cooRowInd[5]=1,	cooRowInd[6]=1,	cooRowInd[7]=2,	cooRowInd[8]=2,	cooRowInd[9]=3;
	cooRowInd[10]=3,	cooRowInd[11]=4,	cooRowInd[12]=4;

	cudaMemcpy(d_cooRowInd, cooRowInd, sizeof(int)*(nnz), cudaMemcpyHostToDevice);
// 1. coo -> csr
	cusparseCreate(&handle);
	cusparseXcoo2csr(handle, d_cooRowInd, nnz,n,d_csrRow, CUSPARSE_INDEX_BASE_ZERO);
	cudaMemcpy(csrRow, d_csrRow,sizeof(int)*(n+1), cudaMemcpyDeviceToHost);
	for(int i=0;i<n+1;++i)
		printf("%d ",csrRow[i]);

	printf("\n");

// 2. csr -> coo
	cudaMemset(d_cooRowInd, 0, sizeof(int)*(nnz));
	memset(cooRowInd,0, sizeof(int)*(nnz));	
	cusparseXcsr2coo(handle, d_csrRow, nnz, n, d_cooRowInd, CUSPARSE_INDEX_BASE_ZERO);
	cudaMemcpy(cooRowInd, d_cooRowInd, sizeof(int)*(nnz), cudaMemcpyDeviceToHost);

	for(int i=0;i<nnz;++i)
		printf("%d ",cooRowInd[i]);
	printf("\n");		

    cusparseDestroy(handle);
	cudaFree(d_cooRowInd), cudaFree(d_csrRow);
	return 0;
}
