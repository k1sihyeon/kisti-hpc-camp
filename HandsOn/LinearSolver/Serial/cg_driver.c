/* cg_driver.c */
#include <stdio.h>
#include <omp.h>
const int MAX_ITER=100000;
const double TOLERANCE=1.0E-8;

void matrix_file_test(void);

void sp_mv(int n, int nnz, int row[nnz], int col[nnz], double a[nnz], double x[n], double b[n]);
void cg(int n, int nnz, int row[nnz], int col[nnz], double a[nnz], double rhs[n], double x[n]);
void convert_sys_to_ge(int nnz, int ge_nnz, int in_row[nnz], int in_col[nnz], double in_a[nnz],
                       int out_row[ge_nnz], int out_col[ge_nnz], double out_a[ge_nnz]);

int main(void)
{
  int selector;
  double start, finish;
  start=omp_get_wtime();
  matrix_file_test();
  finish=omp_get_wtime();
  printf("Elased Time : %lf seconds\n",finish-start);
}



void matrix_file_test(void)
{
  int n,m, nnz,ge_nnz;
  int i, selector;
  FILE *file;
  char temp[128];

  file=fopen("../bcsstk38/bcsstk38.mtx","rt");
  for(i=0;i<13;++i){
    fgets(temp,sizeof(temp),file);
    printf("%d : %s",i, temp);
  }
  fscanf(file,"%d %d %d",&n,&m,&nnz);
  printf("N : %d,  M : %d, NNZ : %d\n",n,m,nnz);

  if(n != m) {
    printf("Matrix is not square matrix\n");
    return;
  }
  
  ge_nnz=2*nnz-n;
  int row[nnz], col[nnz],ge_row[ge_nnz], ge_col[ge_nnz];
  double a[nnz], x[n], b[n], ge_a[ge_nnz];
  for(i=0;i<nnz;++i){
    fscanf(file,"%d %d %lg",&row[i],&col[i],&a[i]);
    row[i] -= 1, col[i] -= 1;
  }
  fclose(file);


  convert_sys_to_ge(nnz, ge_nnz, row, col, a, ge_row, ge_col, ge_a);  
/*
  file=fopen("GEMM.txt","wt");
  for(i=0;i<ge_nnz;i++)
	fprintf(file, "%d %d %lf\n",ge_row[i], ge_col[i], ge_a[i]);
  fclose(file);
*/


  // determine b vector
  for(i=0;i<n;++i)
    x[i]=1.0;
  sp_mv(n,ge_nnz,ge_row,ge_col,ge_a,x,b);


  // initial guess
  for(i=0;i<n;++i)
    x[i]=0.0;

  cg(n,ge_nnz,ge_row,ge_col,ge_a,b,x);
  
  file=fopen("result_x.txt","wt");
  fprintf(file,"%s\n", "# i X ");
  for(i=0;i<n;++i)
    fprintf(file,"%d %lf\n",i,x[i]);

  fclose(file);

}
