/* cg_driver.c */
#include <stdio.h>
#include <mpi.h>
const int MAX_ITER=100000;
//const int MAX_ITER=1000;
const double TOLERANCE=1.0E-8;

void matrix_file_test(const int, const int);
int min(int x, int y);
void para_range(int n1,int n2, int nprocs, int myrank, int *ista, int *iend);

void sp_mv(int n, int nnz, int row[nnz], int col[nnz], double a[nnz], double x[n], double b[n],int myrank, int nprocs, int istart, int iend,int Starts[nprocs], int Ends[nprocs]);

void cg(int n, int nnz, int row[nnz], int col[nnz], double a[nnz], double rhs[n], double x[n],int myrank, int nprocs, int istart, int iend, int Starts[nprocs], int Ends[nprocs]);

void convert_sys_to_ge(int nnz, int ge_nnz, int in_row[nnz], int in_col[nnz], double in_a[nnz],
                       int out_row[ge_nnz], int out_col[ge_nnz], double out_a[ge_nnz]);

int main(void)
{
  int selector;
  double start, finish;
  int myrank, nprocs;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Barrier(MPI_COMM_WORLD);
  start=MPI_Wtime();

  matrix_file_test(myrank, nprocs);

  MPI_Barrier(MPI_COMM_WORLD);
  finish=MPI_Wtime();
  if(myrank==0) printf("Elased Time : %lf seconds\n",finish-start);
  
  MPI_Finalize();
  return 0;  
}



void matrix_file_test(const int myrank, const int nprocs)
{
  int n,m, nnz,ge_nnz;
  int i, selector;
  FILE *file;
  char temp[128];
  MPI_Request ireq[3];

  if(myrank==0){
    printf("Select matrix file'\n");
    printf("\t1 : bcsstk01/bcsstk01.mtx\n");
    printf("\t2 : bcsstk13/bcsstk13.mtx\n");
    printf("\t3 : bcsstk15/bcsstk15.mtx\n");
    printf("\t4 : bcsstk38/bcsstk38.mtx\n");
    printf("(1,2,3,4) : ");
//  scanf("%d",&selector);
    selector=4;

    switch (selector)
    {
      case 1 :  
        file=fopen("../bcsstk01/bcsstk01.mtx","rt");
        break;
      case 2 :
        file=fopen("../bcsstk13/bcsstk13.mtx","rt");
        break;
      case 3 :
        file=fopen("../bcsstk15/bcsstk15.mtx","rt");
        break;
      case 4 :
        file=fopen("../bcsstk38/bcsstk38.mtx","rt");
        break;
      default:
        printf("You must select 1~4\n");
    }
    for(i=0;i<13;++i){
      fgets(temp,sizeof(temp),file);
//      printf("%d : %s",i, temp);
    }
    fscanf(file,"%d %d %d",&n,&m,&nnz);
    printf("N : %d,  M : %d, NNZ : %d\n",n,m,nnz);

    if(n != m) {
      printf("Matrix is not square matrix\n");
      return;
    }
  }
  MPI_Ibcast(&n,   1, MPI_INT, 0, MPI_COMM_WORLD,&ireq[0]);
  MPI_Ibcast(&n,   1, MPI_INT, 0, MPI_COMM_WORLD,&ireq[1]);
  MPI_Ibcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD,&ireq[2]);
  MPI_Waitall(3,ireq,MPI_STATUSES_IGNORE);

  ge_nnz=2*nnz-n;
  int row[nnz], col[nnz],ge_row[ge_nnz], ge_col[ge_nnz];
  double a[nnz], x[n], b[n], ge_a[ge_nnz];

  if(myrank==0){
    for(i=0;i<nnz;++i){
      fscanf(file,"%d %d %lg",&row[i],&col[i],&a[i]);
      row[i] -= 1, col[i] -= 1;
    }
    fclose(file);
    convert_sys_to_ge(nnz, ge_nnz, row, col, a, ge_row, ge_col, ge_a);  
  }

  MPI_Ibcast(ge_row,ge_nnz,MPI_INT, 0, MPI_COMM_WORLD,&ireq[0]);
  MPI_Ibcast(ge_col,ge_nnz,MPI_INT, 0, MPI_COMM_WORLD,&ireq[1]);
  MPI_Ibcast(ge_a,  ge_nnz,MPI_DOUBLE, 0, MPI_COMM_WORLD,&ireq[2]);
  MPI_Waitall(3,ireq,MPI_STATUSES_IGNORE);

  int istart,iend, Starts[nprocs], Ends[nprocs];
  para_range(0,n-1,nprocs,myrank, &istart, &iend);
  Starts[myrank]=istart,  Ends[myrank]=iend;
  for(int i=0;i<nprocs;i++)
  {
    MPI_Bcast(&Starts[i], 1, MPI_INT, i, MPI_COMM_WORLD);
    MPI_Bcast(&Ends[i], 1, MPI_INT, i, MPI_COMM_WORLD);
  }



  // determine b vector
  for(i=0;i<n;++i)
    x[i]=1.0;

  sp_mv(n,ge_nnz,ge_row,ge_col,ge_a,x,b,myrank, nprocs,istart,iend,Starts,Ends);


  // initial guess
  for(i=0;i<n;++i)
    x[i]=0.0;


  cg(n,ge_nnz,ge_row,ge_col,ge_a,b,x,myrank, nprocs, istart,iend, Starts, Ends);

  if(myrank==0){  
    file=fopen("result_x.txt","wt");
    fprintf(file,"%s\n", "# i X ");
    for(i=0;i<n;++i)
      fprintf(file,"%d %lf\n",i,x[i]);
    fclose(file);
  }

}

////////////////////////////////////////
int min(int x, int y){
    int v;
    if (x>=y) v = y;
    else v = x;
    return v;
}

void para_range(int n1,int n2, int nprocs, int myrank, int *ista, int *iend){
   int iwork1, iwork2;
   iwork1 = (n2-n1+1)/nprocs;
   iwork2 = (n2-n1+1) % nprocs;
   *ista= myrank*iwork1 + n1 + min(myrank, iwork2);
   *iend = *ista + iwork1 - 1;
   if(iwork2 > myrank) *iend = *iend + 1;
}

