/* cg.c */
#include <stdio.h>
#include <math.h>
#include <mpi.h>

const int MAX_ITER;
const double TOLERANCE;

void sp_mv_serial(int n, int nnz, int row[nnz], int col[nnz], double a[nnz], double x[n], double b[n])
{
  int i,j,k;
  for(i=0;i<n;++i)
    b[i]=0.0;

  for(k=0;k<nnz;++k)
  {
    i=row[k], j=col[k];
    b[i]=b[i]+a[k]*x[j];
  }
  return;
}

void sp_mv(int n, int nnz, int row[nnz], int col[nnz], double a[nnz], double x[n], double b[n],const int istart, const int iend)
{
  int i,j,k;
  // partial
  for(i=istart;i<=iend;++i)
    b[i]=0.0;

  for(k=0;k<nnz;++k)
  {
    i=row[k];
    if(i>=istart && i<=iend){
      j=col[k];
      b[i]=b[i]+a[k]*x[j];
    }
  }

  return;
}



void innerproduct(int n, double x[n], double y[n], double *inprod,const int istart, const int iend)
{
  int ii;
  double p_inprod=0.0;
  for(ii=istart;ii<=iend;++ii)
    p_inprod += x[ii]*y[ii];

  MPI_Allreduce(&p_inprod, inprod, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
}

void cg(int n, int nnz, int row[nnz], int col[nnz], double a[nnz], double rhs[n], double x[n],const int myrank, const int nprocs,const int istart, const int iend)
{
//  Solve Sparse matrix equation
//  matrix format is general form and storage format is coordinate format
//    N : The order of the matrix
//    NNZ : #of non-zero elements in the matrix
//    row, col : row and column indices of the non-zero elements
//    A : non-zero elements of the matrix
//    B : RHS
//    x : approximate solution vector(INOUT)
  int ii,jj;
  double alpha=0.0, beta=0.0, temp1, temp2, res0tol=0.0;
  double res[n], p[n], ax[n], ap[n];

  int istarts[nprocs],iends[nprocs],temp11[nprocs],temp12[nprocs];
  int cnt;

  for(int i=0;i<nprocs;i++){
    istarts[i]=iends[i]=0;
  }  
  istarts[myrank]=istart, iends[myrank]=iend;
  MPI_Allreduce(istarts,temp11,nprocs,MPI_INT,MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(iends, temp12, nprocs,MPI_INT,MPI_SUM, MPI_COMM_WORLD);
  for(int i=0;i<nprocs;i++){
    istarts[i]=temp11[i], iends[i]=temp12[i];
  }
 
  sp_mv(n,nnz,row,col,a,x,ax,istart,iend);
  
  for(int i=0;i<nprocs;i++)
  {
    cnt=iends[i]-istarts[i]+1;
    MPI_Bcast((double*)&ax[istarts[i]],cnt,MPI_DOUBLE,i,MPI_COMM_WORLD);
  }


  for(ii=0;ii<n;++ii)
  {
    res[ii] = rhs[ii]-ax[ii];
    p[ii]=res[ii];
  }
  innerproduct(n,res,res,&res0tol,istart,iend);
//  MPI_Barrier(MPI_COMM_WORLD);
  if(myrank==0) printf("[CG] conjugate gradient is started.");

  for(ii=1;ii<MAX_ITER;++ii)
  {
    if(myrank==0 && ii%2==0) printf("[CG] mse %12.6e with a tolerance criteria of %12.6e at %10d iterations.\n",sqrt(temp2/res0tol), TOLERANCE, ii);

    innerproduct(n,res,res,&temp1,istart,iend); 
    sp_mv(n,nnz,row,col,a,p,ap,istart,iend);
    for(int i=0;i<nprocs;i++)
    {
      cnt=iends[i]-istarts[i]+1;
      MPI_Bcast((double*)&ap[istarts[i]],cnt,MPI_DOUBLE,i,MPI_COMM_WORLD);
    }

    innerproduct(n,ap,p,&temp2,istart,iend);
    alpha=temp1/temp2;
    for(jj=0;jj<n;++jj)
    {
      x[jj]+= alpha*p[jj];
      res[jj] -= alpha*ap[jj];
    }
    innerproduct(n,res,res,&temp2,istart,iend);
    if(sqrt(temp2/res0tol)<TOLERANCE) break;
    beta=temp2/temp1;
    for(jj=istart;jj<=iend;++jj)
      p[jj]=res[jj]+beta*p[jj];
  }

  if(myrank==0) printf("[CG] Finished with total iteration= %10d, mse= %15.9e\n",ii,sqrt(temp2/res0tol));
  
//  MPI_Barrier(MPI_COMM_WORLD);
}

void convert_sys_to_ge(int nnz, int ge_nnz, int in_row[nnz], int in_col[nnz], double in_a[nnz],
                       int out_row[ge_nnz], int out_col[ge_nnz], double out_a[ge_nnz])
{
  int i, tmp;
  for(i=0;i<nnz;++i)
  {
    out_row[i]=in_row[i];
    out_col[i]=in_col[i];
    out_a[i]=in_a[i];
  }
  tmp=nnz;
  for(i=0;i<nnz;++i)
  {
    if(in_row[i] != in_col[i]){
      out_row[tmp]=in_col[i];
      out_col[tmp]=in_row[i];
      out_a[tmp]=in_a[i];
      tmp++;
    }
  }
}
