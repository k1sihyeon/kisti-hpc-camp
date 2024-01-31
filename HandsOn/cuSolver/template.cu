
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime.h>


int main()
{
// Set up the data
	const int n=5;
	const int nnz=13;
	int csrRow[n+1];
	int csrCol[nnz];
	double csrA[nnz];
	double x[n],b[n];
	csrRow[0]=0,	csrRow[1]=5, 	csrRow[2]=7,	csrRow[3]=9,	csrRow[4]=11,	csrRow[5]=13;

	csrCol[0]=0,	csrCol[1]=1,	csrCol[2]=2,	csrCol[3]=3,	csrCol[4]=4;
    csrCol[5]=0,  	csrCol[6]=1,  	csrCol[7]=0,  	csrCol[8]=2,  	csrCol[9]=0;
    csrCol[10]=3, 	csrCol[11]=0, 	csrCol[12]=4;

	csrA[0]=4.0,	csrA[1]=1.0,	csrA[2]=2.0,	csrA[3]=0.5,	csrA[4]=2.0;
	csrA[5]=1.0,	csrA[6]=0.5,	csrA[7]=2.0,	csrA[8]=3.0,	csrA[9]=0.5;
	csrA[10]=0.625,	csrA[11]=2.0,	csrA[12]=16.0;

	// x={1,2,3,4,5} -> Ax=b 
	// x={1,2,3,4,5} : target solution
	b[0]=24.0,		b[1]=2.0,		b[2]=11.0,		b[3]=3.0,		b[4]=82;

	x[0]=0.0,		x[1]=0.0,		x[2]=0.0,		x[3]=0.0,		x[4]=0.0;

// step 1 : declare vars.
	cusolverSpHandle_t cusolverH=NULL;
	cusparseMatDescr_t descrA=NULL;
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	
	int *d_csrRow, *d_csrCol;
	double *d_csrA, *d_b, *d_x;
	int reorder=0, singularity=0;
	const double tol=1e-8;

    cudaError_t cudaStat1=cudaSuccess;
    cudaError_t cudaStat2=cudaSuccess;
    cudaError_t cudaStat3=cudaSuccess;
    cudaError_t cudaStat4=cudaSuccess;
    cudaError_t cudaStat5=cudaSuccess;

// step 2 : create cusolver handle
	cusolver_status = cusolverSpCreate(&cusolverH);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	cusparse_status = cusparseCreateMatDescr(&descrA);
	assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

	cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);	

// step 3 : allocate memory & copy data
	cudaStat1=cudaMalloc((int**)&d_csrRow, sizeof(int)*(n+1));
	cudaStat2=cudaMalloc((int**)&d_csrCol, sizeof(int)*nnz);
	cudaStat3=cudaMalloc((double**)&d_csrA, sizeof(double)*nnz);
	cudaStat4=cudaMalloc((double**)&d_b, sizeof(double)*n);
	cudaStat5=cudaMalloc((double**)&d_x, sizeof(double)*n);
	assert(cudaStat1==cudaSuccess);
	assert(cudaStat2==cudaSuccess);
	assert(cudaStat3==cudaSuccess);
	assert(cudaStat4==cudaSuccess);
    assert(cudaStat5==cudaSuccess);

	cudaStat1=cudaMemcpy(d_csrRow, csrRow, sizeof(int)*(n+1), cudaMemcpyHostToDevice);
	cudaStat2=cudaMemcpy(d_csrCol, csrCol, sizeof(int)*(nnz), cudaMemcpyHostToDevice);
	cudaStat3=cudaMemcpy(d_csrA, csrA, sizeof(double)*(nnz), cudaMemcpyHostToDevice);
	cudaStat4=cudaMemcpy(d_b, b, sizeof(double)*n, cudaMemcpyHostToDevice);
    assert(cudaStat1==cudaSuccess);
    assert(cudaStat2==cudaSuccess);
    assert(cudaStat3==cudaSuccess);
    assert(cudaStat4==cudaSuccess);


	cusolver_status=cusolverSpDcsrlsvqr(cusolverH, n, nnz, descrA,
									d_csrA, d_csrRow, d_csrCol, d_b, tol, reorder, d_x, &singularity); 
	assert(cusolver_status==CUSOLVER_STATUS_SUCCESS);
	cudaStat4=cudaMemcpy(x, d_x, sizeof(double)*n, cudaMemcpyDeviceToHost);
	assert(cudaStat1==cudaSuccess);

	assert(cusolver_status==CUSOLVER_STATUS_SUCCESS);
	for(int i=0;i<n;++i)
		printf("x: %d - %lf \n",i, x[i]);

	cusparseDestroyMatDescr(descrA);
	cusolverSpDestroy(cusolverH);
	cudaFree(d_csrRow), cudaFree(d_csrCol),	cudaFree(d_csrA),	cudaFree(d_b),	cudaFree(d_x);
	return 0;
}
