/* cg.c */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
const int MAX_ITER;
const double TOLERANCE;

void sp_mv(int n, int nnz, int row[nnz], int col[nnz], double a[nnz], double x[n], double b[n],int myrank, int nprocs, int istart, int iend,int Starts[nprocs], int Ends[nprocs])
{
  int i,j,k;
  int cnt;
  MPI_Request ireq[nprocs];
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


  for(i=0;i<nprocs;i++)
  {
    int cnt=Ends[i]-Starts[i]+1;
    MPI_Ibcast(&b[Starts[i]],cnt, MPI_DOUBLE, i,MPI_COMM_WORLD,&ireq[i]);
  }
  MPI_Waitall(nprocs,ireq,MPI_STATUSES_IGNORE);
  return;
}

void innerproduct(int n, double x[n], double y[n], double *inprod,int istart, int iend)
{
  int ii;
  double p_inprod=0.0;
  for(ii=istart;ii<=iend;++ii)
    p_inprod += x[ii]*y[ii];

  MPI_Allreduce((double*)&p_inprod, inprod, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
}

void cg(int n, int nnz, int row[nnz], int col[nnz], double a[nnz], double rhs[n], double x[n],int myrank, int nprocs, int istart, int iend, int Starts[nprocs], int Ends[nprocs])
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
  MPI_Request ireq[nprocs];
  sp_mv(n,nnz,row,col,a,x,ax,myrank,nprocs,istart,iend,Starts,Ends);
  for(ii=istart;ii<=iend;++ii)
  {
    res[ii] = rhs[ii]-ax[ii];
    p[ii]=res[ii];
  }
  for(int i=0;i<nprocs;i++)
  {
    int cnt=Ends[i]-Starts[i]+1;
    MPI_Ibcast(&p[Starts[i]],cnt, MPI_DOUBLE, i,MPI_COMM_WORLD,&ireq[i]);
  }
  MPI_Waitall(nprocs,ireq,MPI_STATUSES_IGNORE);

  
  innerproduct(n,res,res,&res0tol,istart,iend);
  if(myrank==0) printf("[CG] conjugate gradient is started.");


  for(ii=1;ii<MAX_ITER;++ii)
  {
    if(myrank==0 && ii%20==0) printf("[CG] mse %12.6e with a tolerance criteria of %12.6e at %10d iterations.\n",sqrt(temp2/res0tol), TOLERANCE, ii);

    innerproduct(n,res,res,&temp1,istart,iend); 
    sp_mv(n,nnz,row,col,a,p,ap,myrank,nprocs,istart,iend,Starts,Ends);
    innerproduct(n,ap,p,&temp2,istart,iend);
    alpha=temp1/temp2;

    for(jj=istart;jj<=iend;++jj)
    {
      x[jj]+= alpha*p[jj];
      res[jj] -= alpha*ap[jj];
    }
    innerproduct(n,res,res,&temp2,istart,iend);
    if(sqrt(temp2/res0tol)<TOLERANCE) break;
    beta=temp2/temp1;
    for(jj=istart;jj<=iend;++jj)
      p[jj]=res[jj]+beta*p[jj];

    for(int i=0;i<nprocs;i++)
    {
      int cnt=Ends[i]-Starts[i]+1;
      MPI_Ibcast(&p[Starts[i]],cnt, MPI_DOUBLE, i,MPI_COMM_WORLD,&ireq[i]);
    }
    MPI_Waitall(nprocs,ireq,MPI_STATUSES_IGNORE);
  }

  if(myrank==0){
    for(int i=1;i<nprocs;i++){
      int cnt=Ends[i]-Starts[i]+1;
      MPI_Irecv(&x[Starts[i]],cnt,MPI_DOUBLE,i,i,MPI_COMM_WORLD,&ireq[i]);
    }
  }else{
    int cnt=iend-istart+1;
    MPI_Isend(&x[istart],cnt,MPI_DOUBLE,0,myrank,MPI_COMM_WORLD,&ireq[myrank]);
  }  
  MPI_Waitall(nprocs-1,&ireq[1],MPI_STATUSES_IGNORE);


  if(myrank==0) printf("[CG] Finished with total iteration= %10d, mse= %15.9e\n",ii,sqrt(temp2/res0tol));
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
