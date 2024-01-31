/* cg_driver.c */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <omp.h>

void matrix_file_test(void);

void sp_mv(int n, int nnz, int *row, int *col, double *a, double *x, double *b);
//void cg(int n, int nnz, int row[nnz], int col[nnz], double a[nnz], double rhs[n], double x[n]);
void convert_sys_to_ge(int nnz, int ge_nnz, int *in_row, int *in_col, double *in_a,
                       int *out_row, int *out_col, double *out_a);

int main(void)
{
  double start, finish;
  start=omp_get_wtime();
  matrix_file_test();
  finish=omp_get_wtime();
  printf("Elased Time : %lf seconds\n",finish-start);
}



void matrix_file_test(void)
{
  int n,m, nnz;
  int i;
  FILE *file;
  char temp[128];
//  file=fopen("../e30r5000/e30r5000_sort.txt","rt");  // one-base indexing  
  file=fopen("../bcsstk38/GEMM.txt","rt");			// zero-base indexing ==> 64 line comment out
  fgets(temp,sizeof(temp),file);
  printf("%d : %s",1, temp);

  fscanf(file,"%d %d %d",&n,&m,&nnz);
  printf("N : %d,  M : %d, NNZ : %d\n",n,m,nnz);

  if(n != m) {
    printf("Matrix is not square matrix\n");
    return;
  }



  int *row, *col;
  double *a, *x, *b;

  row=(int*)malloc(sizeof(int)*nnz);
  col=(int*)malloc(sizeof(int)*nnz);

  a=(double*)malloc(sizeof(double)*nnz);
  x=(double*)malloc(sizeof(double)*n);
  b=(double*)malloc(sizeof(double)*n);

  for(i=0;i<nnz;++i){
    fscanf(file,"%d %d %lg",&row[i],&col[i],&a[i]);
//    row[i] -= 1, col[i] -= 1;
  }
  fclose(file);

  // determine b vector
  for(i=0;i<n;++i)
    x[i]=1.0;
  sp_mv(n,nnz,row,col,a,x,b);


  // initial guess
  for(i=0;i<n;++i)
    x[i]=0.0;



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
  cusolver_status=cusolverSpCreate(&cusolverH);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  cusparse_status = cusparseCreateMatDescr(&descrA);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);


  // step 3: allocate memory & copy data
  int *d_cooRowInd;
  cudaStat1 = cudaMalloc((int**)&d_cooRowInd, sizeof(int)*nnz);
  cudaStat2 = cudaMemcpy(d_cooRowInd, row, sizeof(int)*nnz, cudaMemcpyHostToDevice);//Default);
  assert(cudaStat1==cudaSuccess);
  assert(cudaStat2==cudaSuccess);

  cudaStat1 = cudaMalloc((int**)&d_csrRow, sizeof(int)*(n+1));
  cudaStat2 = cudaMalloc((int**)&d_csrCol, sizeof(int)*nnz);
  cudaStat3 = cudaMalloc((double**)&d_csrA, sizeof(double)*nnz);
  cudaStat4 = cudaMalloc((double**)&d_b, sizeof(double)*n);
  cudaStat5 = cudaMalloc((double**)&d_x, sizeof(double)*n);
  assert(cudaStat1==cudaSuccess);
  assert(cudaStat2==cudaSuccess);
  assert(cudaStat3==cudaSuccess);
  assert(cudaStat4==cudaSuccess);
  assert(cudaStat5==cudaSuccess);
 
  // convert coo => csr 
  cusparseHandle_t cusp_H;
  cusparseCreate(&cusp_H);
  cusparseXcoo2csr(cusp_H, d_cooRowInd, nnz, n, d_csrRow, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDestroy(cusp_H);


   
  cudaStat1=cudaMemcpy(d_csrCol, col, sizeof(int)*nnz, cudaMemcpyHostToDevice);//Default);
  cudaStat2=cudaMemcpy(d_csrA, a, sizeof(double)*nnz, cudaMemcpyHostToDevice);//Default);
  cudaStat3=cudaMemcpy(d_b, b, sizeof(double)*n, cudaMemcpyHostToDevice);//Default);
  assert(cudaStat1==cudaSuccess);
  assert(cudaStat2==cudaSuccess);
  assert(cudaStat3==cudaSuccess);

  cusolver_status = cusolverSpDcsrlsvqr(cusolverH, n, nnz, descrA,
									d_csrA, d_csrRow, d_csrCol, d_b, tol, reorder, d_x, &singularity);


  assert(cusolver_status==CUSOLVER_STATUS_SUCCESS);
  cudaStat4=cudaMemcpy(x, d_x, sizeof(double)*n, cudaMemcpyDefault);
  printf("%s\n",cudaGetErrorName(cudaStat4));
  assert(cudaStat4 == cudaSuccess);


  file=fopen("result_x.txt","wt");
  fprintf(file,"%s\n", "# i X ");

  for(i=0;i<n;++i){
    fprintf(file,"%d %lf\n",i,x[i]);
  }
  fclose(file);

  cusparseDestroyMatDescr(descrA);
  cusolverSpDestroy(cusolverH);

  cudaFree(d_csrRow), cudaFree(d_csrCol), cudaFree(d_csrA), cudaFree(d_b), cudaFree(d_x);
  free(row), free(col), free(a), free(x), free(b);

}
