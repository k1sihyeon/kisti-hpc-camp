/* stream.c */
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

const int M=90*3+1, N=60*3+1; // # of grid points starting from 0 in x-, y-direction
//const int M=90*2+1, N=60*2+1; // # of grid points starting from 0 in x-, y-direction
//const int M=90*1+1, N=60*1+1; // # of grid points starting from 0 in x-, y-direction


const double LX=6.0, LY=4.0;  // length of domain along x-, y-direction
//const long int MAX_ITER=100000;	// ori
const long int MAX_ITER=300000;
__constant__ double beta, beta_1;
__device__ double psi_old[N][M];;

template<int col> __global__ void Initial(int N, int M, double *psi_new);
template<int row, int col> __global__ void Jacobi_iter(const int N, const int M, double *psi_new);
template<int col> void pre_post(int N, int M, double *h_psi_new, double dx, double dy);

int main(void)
{
  double dx, dy;
  double *psi_new, h_psi[M*N];
//  int dimx=32, dimy=32;
//  int dimx=1, dimy=N; // ok
  int dimx=M, dimy=1;	// true.

  dx=LX/(M-1),  dy=LY/(N-1);
  double h_beta=(dx/dy);
  double h_beta_1=1.0/(2.0*(1.0+h_beta*h_beta));

  cudaMemcpyToSymbol(beta, &h_beta,sizeof(double));
  cudaMemcpyToSymbol(beta_1, &h_beta_1,sizeof(double));
  cudaMalloc((void**)&psi_new, M*N*sizeof(double));
  dim3 block(dimx,dimy);
  dim3 grid((M+block.x-1)/block.x, (N+block.y-1)/block.y);
  // Initialize problem
  Initial<M><<<grid,block>>>(N, M, psi_new);
  cudaDeviceSynchronize();

  // Jacobi_ieration
  Jacobi_iter<N, M><<<grid,block>>>(N,M,psi_new);
  cudaDeviceSynchronize();
  cudaMemcpy(h_psi,psi_new, M*N*sizeof(double),cudaMemcpyDefault);
  // write mesh & post-processihng file
  pre_post<M>(N, M, h_psi, dx, dy);
  cudaFree(psi_new);
  return 0;
}

template<int col> __global__ void Initial(int N, int M, double *psi_new)
{
  int idx_x=blockDim.x*blockIdx.x + threadIdx.x;
  int idx_y=blockDim.y*blockIdx.y + threadIdx.y;
  double (*pP)[col]=(double (*)[col])psi_new;
  if(idx_x<M && idx_y<N) {
	pP[idx_y][idx_x]=0.0;
  }

  // boundary conditions
  int divide = (int)(M-1)*0.5;
  if(idx_y==N-1 && idx_x<divide) pP[idx_y][idx_x]=0.0;	// bottom(left)
  if(idx_y==N-1 && idx_x>=divide && idx_x<M) pP[idx_y][idx_x]=100.0;	//bottom(right)
  if(idx_x==0 && idx_y<N) pP[idx_y][idx_x]=0.0;	// left wall
  if(idx_y==0 && idx_x<M) pP[idx_y][idx_x]=0.0;	// upper wall
}

/**********************************************************************************/
template<int row, int col> __global__ void Jacobi_iter(const int N, const int M, double *psi_new)
{
  long int iter;
  double error=1.0;
  
  int idx_x=blockDim.x*blockIdx.x + threadIdx.x;
  int idx_y=blockDim.y*blockIdx.y + threadIdx.y;
  double (*pP)[col]=(double (*)[col])psi_new;
  for(iter=1;iter<MAX_ITER;++iter)
  {
    if(idx_x<M && idx_y<N) {
		psi_old[idx_y][idx_x]=pP[idx_y][idx_x];
	}
	__threadfence();
   __syncthreads();


	if(idx_x > 0 &&idx_x<M-1){
		if(idx_y> 0 && idx_y<N-1){
        pP[idx_y][idx_x]=beta_1*(psi_old[idx_y][idx_x+1]+psi_old[idx_y][idx_x-1]+
                             beta*beta* (psi_old[idx_y+1][idx_x]+psi_old[idx_y-1][idx_x]));
		}
	}
	__threadfence();
   __syncthreads();

    // Right Neumann Boundary Condition
	if(idx_x==M-1 && idx_y<N) pP[idx_y][idx_x]=pP[idx_y][idx_x-1];
	__threadfence();
	__syncthreads();

	if(idx_y==0 && idx_x==0 && iter%1000==0) printf("Iteration = %ld\n", iter);
	__threadfence();
	__syncthreads();

  }
}

/***********************************************************************/

template<int col> void pre_post(const int N, const int M, double *h_psi_new, double dx, double dy)
{
  int i,j;
  double coord_x[M][N], coord_y[M][N];
  double (*psi_new)[col]=(double (*)[col])h_psi_new;

  double temp_x, temp_y;
  int num_node, num_ele;
  struct node_info
  {
    int id, node_i, node_j;
  };
  struct ele_info
  {
    int id, node1, node2, node3, node4;
  };

  struct node_info node_id[M*N];
  struct ele_info ele_id[(M-1)*(N-1)];
  int n1, n2, n3, n4;
  double x1,y1,x2,y2,x3,y3,x4,y4;
  double psi1, psi2, psi3, psi4;

  for(i=1;i<=M;++i)
  {
    temp_x=(double)((i-1))*dx;
    for(j=1;j<=N;++j)
    {
      temp_y=(double)(LY-(j-1))*dy;
      coord_x[i-1][j-1]=temp_x;
      coord_y[i-1][j-1]=temp_y;
    }
  }
  
  // write mesh file
  FILE *mesh=fopen("stream_cuda.msh","wt");
  fprintf(mesh,"%s\n","$MeshFormat");
  fprintf(mesh,"%s\n","2.2 0 8");
  fprintf(mesh,"%s\n","$EndMeshFormat");
  fprintf(mesh,"%s\n","$Nodes");
  fprintf(mesh,"%d\n",M*N);

  num_node=0,   num_ele=0;
  for(j=1;j<=N;++j)
    for(i=1;i<=M;++i)
    {
      num_node++;
      node_id[num_node-1].id=num_node;
      node_id[num_node-1].node_i=i;
      node_id[num_node-1].node_j=j;
      fprintf(mesh,"%10d %10.5f %10.5f %10.5f\n",num_node,coord_x[i-1][j-1],coord_y[i-1][j-1],0.0);
    }

  fprintf(mesh,"%s\n","$EndNodes");
  fprintf(mesh,"%s\n","$Elements");
  fprintf(mesh,"%d\n",(M-1)*(N-1));

  for(j=1;j<=N-1;++j)
    for(i=1;i<=M-1;++i)
    {
      num_ele++;
      ele_id[num_ele-1].id=num_ele;
      ele_id[num_ele-1].node1=num_ele+(j-1);
      ele_id[num_ele-1].node2=ele_id[num_ele-1].node1+M;
      ele_id[num_ele-1].node3=ele_id[num_ele-1].node2+1;
      ele_id[num_ele-1].node4=ele_id[num_ele-1].node1+1;
      fprintf(mesh,"%d 3 2 0 1 %d %d %d %d\n",ele_id[num_ele-1].id,ele_id[num_ele-1].node1,
              ele_id[num_ele-1].node2, ele_id[num_ele-1].node3, ele_id[num_ele-1].node4);

    }

  fprintf(mesh,"%s\n","$EndElements");
  fclose(mesh);

  FILE *post=fopen("stream_cuda.pos","wt");
  fprintf(post,"%s\n","// View Stream function ");
  fprintf(post,"%s\n","View \"Strem function(Psi)\" {");
  for(i=1;i<=num_ele;++i)
  {
    n1=ele_id[i-1].node1, n2=ele_id[i-1].node2, n3=ele_id[i-1].node3, n4=ele_id[i-1].node4;

    x1=coord_x[node_id[n1-1].node_i-1][node_id[n1-1].node_j-1]; 
    y1=coord_y[node_id[n1-1].node_i-1][node_id[n1-1].node_j-1];

    x2=coord_x[node_id[n2-1].node_i-1][node_id[n2-1].node_j-1];
    y2=coord_y[node_id[n2-1].node_i-1][node_id[n2-1].node_j-1];

    x3=coord_x[node_id[n3-1].node_i-1][node_id[n3-1].node_j-1];
    y3=coord_y[node_id[n3-1].node_i-1][node_id[n3-1].node_j-1];

    x4=coord_x[node_id[n4-1].node_i-1][node_id[n4-1].node_j-1];
    y4=coord_y[node_id[n4-1].node_i-1][node_id[n4-1].node_j-1];

    fprintf(post,"SQ(");
    fprintf(post,"%10.5f, %10.5f, 0.0, ",x1,y1);
    fprintf(post,"%10.5f, %10.5f, 0.0, ",x2,y2);
    fprintf(post,"%10.5f, %10.5f, 0.0, ",x3,y3);
    fprintf(post,"%10.5f, %10.5f, 0.0 )",x4,y4);
    fprintf(post," {");
    psi1=psi_new[node_id[n1-1].node_j-1][node_id[n1-1].node_i-1];
    psi2=psi_new[node_id[n2-1].node_j-1][node_id[n2-1].node_i-1];
    psi3=psi_new[node_id[n3-1].node_j-1][node_id[n3-1].node_i-1];
    psi4=psi_new[node_id[n4-1].node_j-1][node_id[n4-1].node_i-1];
    fprintf(post,"%10.5f, %10.5f, %10.5f, %10.5f",psi1,psi2,psi3,psi4);
    fprintf(post,"};\n");
  }
  fprintf(post,"};");
  fclose(post);

  return;
}

