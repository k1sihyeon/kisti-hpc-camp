/* stream.c */
#include <stdio.h>
#include <math.h>
#include <omp.h>
const double LX=6.0, LY=4.0;  // length of domain along x-, y-direction
const double EPSILON=1e-8;    // tolerance
//const long int MAX_ITER=1000000000;
const long int MAX_ITER=300000;

const int M=90*1+1, N=60*1+1; // # of grid points starting from 0 in x-, y-direction

void Jacobi_iter(int N, int M, double psi_new[N][M],double beta,double beta_1);
void pre_post(int N, int M, double psi_new[N][M], double dx, double dy);

int main(void)
{
  
  double dx, dy, beta, beta_1;
  int i, j, divide;
  double psi_new[N][M];
  double STime, ETime;

  dx=LX/(M-1),  dy=LY/(N-1);
  beta=(dx/dy);
  beta_1=1.0/(2.0*(1.0+beta*beta));
  
  for(i=0;i<N;++i)
    for(j=0;j<M;++j)
    {
      psi_new[i][j]=0.0;
    }
  
  // boundary conditions
  divide = (int)(M-1)*0.5;
  for(i=0;i<divide;++i)
    psi_new[N-1][i]=0.0; // bottom(left)
  for(i=divide;i<M;++i)
    psi_new[N-1][i]=100.0;  // bottom(right)
  for(i=0;i<N;++i)
    psi_new[i][0]=0.0;  // left wall
  for(i=0;i<M;++i)
    psi_new[0][i]=0.0;  // upper wall

  // Jacobi_ieration
  STime=omp_get_wtime();
  Jacobi_iter(N,M,psi_new,beta,beta_1);
  ETime=omp_get_wtime();
  printf("Elapsed Time : %lf sec.\n",ETime-STime);

  // write mesh & post-processihng file
  pre_post(N, M, psi_new, dx, dy);
  return 0;
}

/**********************************************************************************/
void Jacobi_iter(int N, int M,double psi_new[N][M],double beta,double beta_1)
{
  long int iter;
  int i, j;
  double psi_old[N][M];

  for(iter=1;iter<MAX_ITER;++iter)
  {
    for(i=0;i<M;++i)
      for(j=0;j<N;++j)
        psi_old[j][i]=psi_new[j][i];

    for(i=1;i<M-1;++i)
      for(j=1;j<N-1;++j)
        psi_new[j][i]=beta_1*(psi_old[j][i+1]+psi_old[j][i-1]+
                              beta*beta*(psi_old[j+1][i]+psi_old[j-1][i]));


    // Right Neumann Boundary Condition
    for(j=0;j<N;++j)
      psi_new[j][M-1]=psi_new[j][M-2];

	if(iter%1000==0) printf("Iteration =%ld\n",iter);
  }
}

/***********************************************************************/
void pre_post(int N, int M, double psi_new[N][M], double dx, double dy)
{
  int i,j;
  double coord_x[M][N], coord_y[M][N];
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
  FILE *mesh=fopen("stream_serial.msh","wt");
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

  FILE *post=fopen("stream_serial.pos","wt");
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

