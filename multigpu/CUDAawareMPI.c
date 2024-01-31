#include <stdio.h>
#include "mpi.h"
#include "cuda_runtime.h"
#include "string.h"

extern void kernel_wrapper(int *buf, int N);

int main(void)
{
	int myrank, i, N=5;
	int *d_src, h_src[N];
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	cudaSetDevice(myrank);
	int cur_dev;
	cudaGetDevice(&cur_dev);
	cudaMalloc((void**)&d_src,N*sizeof(int));
	printf("GPU : %d, d_src: \n",cur_dev);

// prepare GPU_data
	if(myrank==0){
		for(int i=0;i<N;i++) h_src[i]=i+1;
	}
	else if(myrank==1){
		for(int i=0;i<N;i++) h_src[i]=(i+1)*10;
	}
	cudaMemcpy(d_src, h_src,N*sizeof(int),cudaMemcpyDefault);
	kernel_wrapper(d_src,N);
	if(myrank==0) printf("================================\n");
	MPI_Barrier(MPI_COMM_WORLD);
////////////////////////////

	int *d_rcv;
	MPI_Request ReqS, ReqR;
	cudaMalloc((void**)&d_rcv,N*sizeof(int));

	if(myrank==0){
		MPI_Isend(d_src,N,MPI_INT,1,10,MPI_COMM_WORLD,&ReqS);
		MPI_Irecv(d_rcv,N,MPI_INT,1,100,MPI_COMM_WORLD,&ReqR);
	}
	else if(myrank==1){
		MPI_Isend(d_src,N,MPI_INT,0,100,MPI_COMM_WORLD,&ReqS);
		MPI_Irecv(d_rcv,N,MPI_INT,0,10,MPI_COMM_WORLD,&ReqR);
	}	
	MPI_Wait(&ReqS,MPI_STATUS_IGNORE);
	MPI_Wait(&ReqR,MPI_STATUS_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);	

	printf("GPU : %d, d_rcv: \n",cur_dev);
	kernel_wrapper(d_rcv,N);

	cudaFree(d_src), cudaFree(d_rcv);
	MPI_Finalize();

	return 0;
}
