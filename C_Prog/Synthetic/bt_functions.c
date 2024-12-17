#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "bt_functions.h"
#include "mt64.h"
#include "my_sort.h"



// void read_index_file (char *filename, struct hypergraph *G, char **names)
// {
  
//   int i, q;
//   float score;
//   FILE *f;

//   G->N = 0;
//   f = fopen(filename,"r");
//   while(!feof(f))
//     {
//       q = fscanf(f,"%d %s",&i, score);
//       if (q<=0) goto exit_file_A;
//       if(i>G->N) G->N = i;
//     }
//  exit_file_A:
//   fclose(f);

  
//   //names = (char **)malloc((G->N+1)*sizeof(char *));
//   //for(i=1;i<=G->N;i++) names[i] = (char *)malloc(100*sizeof(char));


//   f = fopen(filename,"r");
//   while(!feof(f))
//     {
//       q = fscanf(f,"%d %s",&i,name);
//       if (q<=0) goto exit_file_B;
//       //sprintf(names[i], "%s", name);
//     }
//  exit_file_B:
//   fclose(f);


  
// }

void read_index_file(char *filename, struct hypergraph *G) {
    int i, q;
    double score;
    FILE *f;

    G->N = 0;

    // Open the file for the first pass to determine the maximum index
    f = fopen(filename, "r");
    if (f == NULL) {
        perror("Error opening file");
        return;
    }

    // Determine the largest index
    while (fscanf(f, "%d %lf", &i, &score) == 2) {
        if (i > G->N) G->N = i;
    }

    fclose(f);

    // Allocate memory for G->pi_values
    G->pi_values = (double *)malloc((G->N+1)*sizeof(double));
    if (G->pi_values == NULL) {
        perror("Error allocating memory for G->pi_values");
        return;
    }

    // Initialize pi_values to 0 (optional, for safety)
    for (i = 0; i <= G->N; i++) {
        G->pi_values[i] = 0.0l;
    }

    // Open the file for the second pass to populate pi_values
    f = fopen(filename, "r");
    if (f == NULL) {
        perror("Error opening file for second pass");
        free(G->pi_values); // Free the allocated memory on failure
        return;
    }

    // Populate the pi_values array
    while (fscanf(f, "%d %lf", &i, &score) == 2) {
        G->pi_values[i] = score;
    }

    fclose(f);
}



void read_data_file (char *filename, struct hypergraph *G)
{
  
  int i, k, j, q, m;
  FILE *f;

  G->M = 0;
  f = fopen(filename,"r");
  while(!feof(f))
    {
      q = fscanf(f,"%d",&k);
      if (q<=0) goto exit_file_A;
      for(j=1;j<=k;j++)
	{
	  q = fscanf(f,"%d",&i);
	  if (q<=0) goto exit_file_A;
	}
      G->M += 1;
    }
 exit_file_A:
  fclose(f);


  G->hyperedges = (int **)malloc((G->M+1)*sizeof(int *));
  // G->pi_values = (double *)malloc((G->N+1)*sizeof(double));
  G->hyperbonds = (int **)malloc((G->N+1)*sizeof(int *));
  G->node_rank = (int **)malloc((G->N+1)*sizeof(int *));

  m = 0;
  f = fopen(filename,"r");
  while(!feof(f))
    {
      q = fscanf(f,"%d",&k);
      if (q<=0) goto exit_file_B;
      m += 1;
      G->hyperedges[m] = (int *)malloc((k+1)*sizeof(int));
      G->hyperedges[m][0] = k;
      for(j=1;j<=k;j++) 
	{
	  q = fscanf(f,"%d",&i);
	  if (q<=0) goto exit_file_B;
	  G->hyperedges[m][j] = i;
	}
    }
 exit_file_B:
  fclose(f);


  //printf("#M = %d\n",G->M); fflush(stdout);

  create_hyperbonds_from_hyperedges (G);
  // generate_pi_values(G);


}




/////////////////////////////////////////////

void deallocate_memory (struct hypergraph *G)
{

  int i;

  for(i=1;i<=G->M;i++) free(G->hyperedges[i]);
  free(G->hyperedges);
  
  free(G->pi_values);

  for(i=0;i<=G->N;i++) free(G->node_rank[i]);
  free(G->node_rank);
  for(i=0;i<=G->N;i++) free(G->hyperbonds[i]);
  free(G->hyperbonds);

  
  free(G);

}


void print_hypergraph(struct hypergraph *G)
{
  int i, j;

  printf("#Pi\n");
  for(i=1;i<=G->N;i++) printf("%d %g\n",i,G->pi_values[i]);
  fflush(stdout);

  printf("#Hyperedges\n");
  for(i=1;i<=G->M;i++)
    {
      printf("%d\t",G->hyperedges[i][0]);
      for(j=1;j<=G->hyperedges[i][0];j++) printf("%d ",G->hyperedges[i][j]);
      printf("\n");
      fflush(stdout);
    }

  printf("#Hyperbonds\n");
  for(i=1;i<=G->N;i++)
    {
      printf("%d %d\t",i,G->node_rank[0][i]);
      for(j=1;j<=G->node_rank[0][i];j++) printf("%d %d\t",G->node_rank[i][j],G->hyperbonds[i][j]);
      printf("\n");
      fflush(stdout);
    }

}


void normalize_pi_values (struct hypergraph *G)
{
  int i;
  G->pi_values[0] = 0.0; 
  for(i=1;i<=G->N;i++)  G->pi_values[0] += log(G->pi_values[i]);
  G->pi_values[0] = exp(G->pi_values[0]/(double)G->N);
  for(i=1;i<=G->N;i++) G->pi_values[i] = G->pi_values[i] / G->pi_values[0];
}



void create_hyperbonds_from_hyperedges (struct hypergraph* G)
{
  int m, s, i, j;

  //
  G->node_rank[0] = (int *)malloc((G->N+1)*sizeof(int));
   G->hyperbonds[0] = (int *)malloc((G->N+1)*sizeof(int));
  for(i=1;i<=G->N;i++) G->node_rank[0][i] = 0;
  for(m=1;m<=G->M;m++)
    {
    for(i=1;i<=G->hyperedges[m][0];i++)
      {
	j = G->hyperedges[m][i];
	G->node_rank[0][j] += 1;
      }
    }
  for(i=1;i<=G->N;i++)
    {
      G->node_rank[i] = (int *)malloc((G->node_rank[0][i] +1)*sizeof(int));
      G->hyperbonds[i] = (int *)malloc((G->node_rank[0][i] +1)*sizeof(int));
      G->hyperbonds[0][i] = G->node_rank[0][i]; 
    }

      
  for(i=1;i<=G->N;i++) G->node_rank[0][i] = 0;
   for(m=1;m<=G->M;m++)
     {
       for(i=1;i<=G->hyperedges[m][0];i++)
	 {
	   j = G->hyperedges[m][i];
	   G->node_rank[0][j] += 1;
	   G->node_rank[j][G->node_rank[0][j]] = i;
	   G->hyperbonds[j][G->node_rank[0][j]] = m;
	 }
     }
   //
  
  
  
}




//pdf p / (1 + p)^2
//cdf 1 / ( 1+ p)
// 1 / (1+p) = e
// p = 1/e - 1

//to be verified
double random_number_from_logistic (void)
{
  return 1.0/genrand64_real3() - 1.0; 
}



void generate_pi_values(struct hypergraph *G)
{
  int i;
  for(i=1;i<=G->N;i++) G->pi_values[i] = random_number_from_logistic();
  return;
}


void compute_probability_model (struct hypergraph *G)
{

  int i, j, m;
  double tmp;
  
  //likelihood
  G->likelihood_ho = G->likelihood_leader = 0.0;
  for(m=1;m<=G->M;m++)
    {

      tmp = 0.0;
      for(i=1;i<=G->hyperedges[m][0];i++) tmp += G->pi_values[G->hyperedges[m][i]];

      G->likelihood_leader += log(G->pi_values[G->hyperedges[m][1]]) - log(tmp);       

      for(i=1;i<=G->hyperedges[m][0]-1;i++)
      {
	for(j=i+1;j<=G->hyperedges[m][0];j++)
	  {
	    G->likelihood_ho += log(G->pi_values[G->hyperedges[m][i]]) - log(tmp); 
	  }
	tmp -= G->pi_values[G->hyperedges[m][i]]; 
      }
  
    }

  G->likelihood_leader /= (double)G->M;
  G->likelihood_ho /= (double)G->M;

  //prior
  G->prior = 0.0;
  for(i=1;i<=G->N;i++) G->prior += log(G->pi_values[i]) - 2.0 * log(1.0 + G->pi_values[i]);

  G->prior /= (double)G->N;

}


void generate_ho_model (int N, int M, int K1, int K2, struct hypergraph *G)
{
  int i, m, K;
  int **control = (int **)malloc(2*sizeof(int *));
  control[0] = (int *)malloc((N+1)*sizeof(int));
  control[1] = (int *)malloc((N+1)*sizeof(int));
  for(i=0;i<=N;i++) control[0][i] = control[1][i] = -1;
  

  G->N = N;
  G->M = M;

  G->hyperedges = (int **)malloc((M+1)*sizeof(int *));
  G->pi_values = (double *)malloc((N+1)*sizeof(double));
  G->hyperbonds = (int **)malloc((N+1)*sizeof(int *));
  G->node_rank = (int **)malloc((N+1)*sizeof(int *));
  

  //
  generate_pi_values(G);
  normalize_pi_values (G);
  //

  
 
  for(m=1;m<=M;m++)
    {

      K = K1 + (int)(genrand64_real3() * (double)(K2-K1) + 0.5);
      if (K > K2) K = K1;

       G->hyperedges[m] = (int *)malloc((K+1)*sizeof(int));
       G->hyperedges[m][0] = K;

       create_hyperedge_ho_model (G, m, control);
      
    }





  create_hyperbonds_from_hyperedges (G);
  


  free(control[0]);
  free(control[1]);
  free(control);
  
  return;
}



void create_hyperedge_ho_model (struct hypergraph *G, int m, int **control)
{
  int i, n;
  double p, norm;
  control[0][0] = 0;

  while(control[0][0] < G->hyperedges[m][0])
    {
      i = (int)((double)G->N * genrand64_real3()) + 1;
      if (i>G->N) i = 1;
      while(control[1][i]>0)
	{
	  i += 1;
	  if (i>G->N) i = 1;
	}

      control[0][0] += 1;
      control[0][control[0][0]] = i;
      control[1][i] = 1;
    }

  //printf("%d\t",control[0][0]);
  //for(i=1;i<=control[0][0];i++) printf("%d ",control[0][i]);
  //printf("\n");

  G->hyperedges[m][0] = 0;
  while(control[0][0] >0)
    {

      if(control[0][0]>1)
	{
      
	  norm = 0.0;
	  for(i=1;i<=control[0][0];i++){
	    norm += G->pi_values[control[0][i]];
	  }
      
	  p = genrand64_real3() * norm;

      
	  norm = 0.0;
	  i = 0;
	  while (p>=norm) {
	    i += 1;
	    norm += G->pi_values[control[0][i]];
	  }
	}

      else i = 1;


      n = control[0][i];
      control[1][n] = -1;
      control[0][i] = control[0][control[0][0]];
      control[0][0] -= 1;


      G->hyperedges[m][0] += 1;
      G->hyperedges[m][G->hyperedges[m][0]] = n;



      //printf("-> %d\t%d\n",i,n);
      //printf("%d\t",control[0][0]);
      //for(i=1;i<=control[0][0];i++) printf("%d ",control[0][i]);
      //printf("\n\n");
      
    }

  //printf("\n\n");

}



///////////////////////////////////////

void generate_leadership_model (int N, int M, int K1, int K2, struct hypergraph *G)
{
  int i, m, K;
  int **control = (int **)malloc(2*sizeof(int *));
  control[0] = (int *)malloc((N+1)*sizeof(int));
  control[1] = (int *)malloc((N+1)*sizeof(int));
  for(i=0;i<=N;i++) control[0][i] = control[1][i] = -1;
  

  G->N = N;
  G->M = M;

  G->hyperedges = (int **)malloc((M+1)*sizeof(int *));
  G->pi_values = (double *)malloc((N+1)*sizeof(double));
  G->hyperbonds = (int **)malloc((N+1)*sizeof(int *));
  G->node_rank = (int **)malloc((N+1)*sizeof(int *));
  

  //
  generate_pi_values(G);
  normalize_pi_values (G);
  //

  
 
  for(m=1;m<=M;m++)
    {

      K = K1 + (int)(genrand64_real3() * (double)(K2-K1) + 0.5);
      if (K > K2) K = K1;

       G->hyperedges[m] = (int *)malloc((K+1)*sizeof(int));
       G->hyperedges[m][0] = K;

       create_hyperedge_leadership_model (G, m, control);
      
    }




  create_hyperbonds_from_hyperedges (G);
  


  free(control[0]);
  free(control[1]);
  free(control);
  
  return;
}



void create_hyperedge_leadership_model (struct hypergraph *G, int m, int **control)
{
  int i, n;
  double p, norm;
  control[0][0] = 0;

  while(control[0][0] < G->hyperedges[m][0])
    {
      i = (int)((double)G->N * genrand64_real3()) + 1;
      if (i>G->N) i = 1;
      while(control[1][i]>0)
	{
	  i += 1;
	  if (i>G->N) i = 1;
	}

      control[0][0] += 1;
      control[0][control[0][0]] = i;
      control[1][i] = 1;
    }

  //printf("%d\t",control[0][0]);
  //for(i=1;i<=control[0][0];i++) printf("%d ",control[0][i]);
  //printf("\n");

  G->hyperedges[m][0] = 0;
  while(control[0][0] >0)
    {

      if(control[0][0]>1)
	{

	  if (G->hyperedges[m][0] == 0)
	    {
	  
	      norm = 0.0;
	      for(i=1;i<=control[0][0];i++){
		norm += G->pi_values[control[0][i]];
	      }
	      
	      p = genrand64_real3() * norm;
	      
	      
	      norm = 0.0;
	      i = 0;
	      while (p>=norm) {
		i += 1;
		norm += G->pi_values[control[0][i]];
	      }
	    }

	  else{
	    i = (int)(genrand64_real3() * (double)control[0][0]) + 1;
	    if (i > control[0][0]) i = 1;
	  }
	  
	}

      else i = 1;


      n = control[0][i];
      control[1][n] = -1;
      control[0][i] = control[0][control[0][0]];
      control[0][0] -= 1;


      G->hyperedges[m][0] += 1;
      G->hyperedges[m][G->hyperedges[m][0]] = n;



      //printf("-> %d\t%d\n",i,n);
      //printf("%d\t",control[0][0]);
      //for(i=1;i<=control[0][0];i++) printf("%d ",control[0][i]);
      //printf("\n\n");
      
    }

  //printf("\n\n");

}




////////////////////////////////////////
////////////////////////////////////////


void evaluate_results (struct hypergraph *G, struct model_results *R)
{
  int m,i,j;
  double tmp;



  //error
  R->error = R->log_error = R->av_error= 0.0;
  
  for(i=1;i<=G->N;i++)
    {
      R->error += (G->pi_values[i]-R->scores[i])*(G->pi_values[i]-R->scores[i]);
      R->log_error += (log(G->pi_values[i])-log(R->scores[i]))*(log(G->pi_values[i])-log(R->scores[i]));
      R->av_error += (G->pi_values[i]/(1.0 + G->pi_values[i]) - R->scores[i]/(1.0 + R->scores[i]))*(G->pi_values[i]/(1.0 + G->pi_values[i]) - R->scores[i]/(1.0 + R->scores[i]));
    }

  R->error = R->error / (double)R->N;
  R->error = sqrt(R->error);
  R->log_error = R->log_error / (double)R->N;
  R->log_error = sqrt(R->log_error);
  R->av_error = R->av_error / (double)R->N;
  R->av_error = sqrt(R->av_error);



  //likelihood
  R->likelihood_ho = R->likelihood_leader = 0.0;
  for(m=1;m<=G->M;m++)
    {

      tmp = 0.0;
      for(i=1;i<=G->hyperedges[m][0];i++) tmp += R->scores[G->hyperedges[m][i]];

      R->likelihood_leader += log(R->scores[G->hyperedges[m][1]]) - log(tmp);       

      for(i=1;i<=G->hyperedges[m][0]-1;i++)
      {
	for(j=i+1;j<=G->hyperedges[m][0];j++)
	  {
	    R->likelihood_ho += log(R->scores[G->hyperedges[m][i]]) - log(tmp); 
	  }
	tmp -= R->scores[G->hyperedges[m][i]]; 
	
      }
  
    }

  R->likelihood_leader /= (double)G->M;
  R->likelihood_ho /= (double)G->M;
  

  //prior
  R->prior = 0.0;
  for(i=1;i<=G->N;i++) R->prior += log(R->scores[i]) - 2.0 * log(1.0 + R->scores[i]);

  R->prior /= (double)G->N;


  //spearman correlation
  R->spearman =  spearman_correlation (G->N, G->pi_values, R->scores);
  //kendall correlation
  R->kendall = kendall_correlation (G->N, G->pi_values, R->scores);
}



double spearman_correlation (int N, double *vec1, double *vec2)
{
  int i, r;
  double val, m1, m2, s1, s2, m12;
  int *idx1 = (int *)malloc((N+1)*sizeof(int));
  int *idx2 = (int *)malloc((N+1)*sizeof(int));
  double *tmp_vec1 = (double *)malloc((N+1)*sizeof(double));
  double *tmp_vec2 = (double *)malloc((N+1)*sizeof(double));
  int *rank1 = (int *)malloc((N+1)*sizeof(int));
  int *rank2 = (int *)malloc((N+1)*sizeof(int));
  
  for(i=1;i<=N;i++)
    {
      idx1[i] = idx2[i] = i;
      tmp_vec1[i] = vec1[i];
      tmp_vec2[i] = vec2[i];
    }

  q_sort_double_with_indx(tmp_vec1,idx1,1,N);
  q_sort_double_with_indx(tmp_vec2,idx2,1,N);


  
  r = 1;
  val = tmp_vec1[N];
  rank1[idx1[N]] = r;
  for(i=N-1;i>=1;i--)
    {
      if(tmp_vec1[i]< val)
	{
	  val = tmp_vec1[i];
	  r += 1;
	}
      rank1[idx1[i]] = r;
    }

  //printf("#A\n");
  //for(i=1;i<=N;i++)  printf("%d %d %g\n",rank1[idx1[i]],idx1[i],tmp_vec1[i]);
  //printf("\n");

  
  r = 1;
  val = tmp_vec2[N];
  rank2[idx2[N]] = r;
  for(i=N-1;i>=1;i--)
    {
      if(tmp_vec2[i]< val)
	{
	  val = tmp_vec2[i];
	  r += 1;
	}
      rank2[idx2[i]] = r;
    }

  //printf("#B\n");
  //for(i=1;i<=N;i++)  printf("%d %d %g\n",rank2[idx2[i]],idx2[i],tmp_vec2[i]);
  //printf("\n");
  

  val = 0.0;
  for(i=1;i<=N;i++) val += (double)(rank1[i]-rank2[i])*(double)(rank1[i]-rank2[i]);
  val = 1.0 - 6.0 *val / ((double)N*((double)N*(double)N -1.0));


  //more general formula
  m12 = m1 = m2 = s1 = s2 = 0.0;
  for(i=1;i<=N;i++)
    {
      m1 += (double)rank1[i];
      m2 += (double)rank2[i];
      s1 += (double)rank1[i]*(double)rank1[i];
      s2 += (double)rank2[i]*(double)rank2[i];
    }

  m1 /= (double)N;
  m2 /= (double)N;
  s1 /= (double)N;
  s2 /= (double)N;
  s1 = s1 - m1*m1;
  s2 = s2 - m2*m2;
  s1 = sqrt(s1);
  s2 = sqrt(s2);

  for(i=1;i<=N;i++) m12 += ((double)rank1[i] - m1)*((double)rank2[i] - m2);
  m12 /= (double)N;

  //printf("%g %g\n",val,m12/s1/s2);
  val = m12 / s1 / s2;

  free(idx1);
  free(idx2);
  free(tmp_vec1);
  free(tmp_vec2);
  free(rank1);
  free(rank2);


  return val;
}




double kendall_correlation (int N, double *vec1, double *vec2)
{
  int i, j;
  double s1, s2, val = 0.0;


  for(i=1;i<N;i++)
    {
      for(j=i+1;j<=N;j++)
	{
	  s1 = 1.0;
	  if (vec1[i] < vec1[j]) s1 = -1.0;
	  s2 = 1.0;
	  if (vec2[i] < vec2[j]) s2 = -1.0;

	  val += s1*s2;
	}
    }

  return 2.0*val/(double)N/(double)(N-1);
}



////////////////////////////////////////
void binarize_ho_model (struct hypergraph *G, struct  hypergraph *H)
{

  int m, i, j;
  
  H->N = G->N;

 

  H->pi_values = (double *)malloc((H->N+1)*sizeof(double));
  for(i=1;i<=H->N;i++) H->pi_values[i] = G->pi_values[i];

  
  H->M = 0;
  for(m=1;m<=G->M;m++)  H->M += (G->hyperedges[m][0]-1)*G->hyperedges[m][0]/2;
 

  H->hyperedges = (int **)malloc((H->M+1)*sizeof(int *));
  for(m=1;m<=H->M;m++) H->hyperedges[m] = (int *)malloc(3*sizeof(int));

  //printf("#%d\n",H->M);
  
  
  H->M = 0;
  for(m=1;m<=G->M;m++)
    {
      for(i=1;i<=G->hyperedges[m][0]-1;i++)
	{
	  for(j=i+1;j<=G->hyperedges[m][0];j++)
	    {
	      H->M += 1;
	      H->hyperedges[H->M][0] = 2;
	      H->hyperedges[H->M][1] = G->hyperedges[m][i];
	      H->hyperedges[H->M][2] = G->hyperedges[m][j];
	    }
	}
    }
  //printf("#%d\n",H->M);
  
  H->hyperbonds = (int **)malloc((H->N+1)*sizeof(int *));
  H->node_rank = (int **)malloc((H->N+1)*sizeof(int *));

  create_hyperbonds_from_hyperedges (H);

}




////////////////////////////////////////
void binarize_leadership_model (struct hypergraph *G, struct  hypergraph *H)
{

  int m, i, j;
  
  H->N = G->N;

 

  H->pi_values = (double *)malloc((H->N+1)*sizeof(double));
  for(i=1;i<=H->N;i++) H->pi_values[i] = G->pi_values[i];

  
  H->M = 0;
  for(m=1;m<=G->M;m++)  H->M += G->hyperedges[m][0]-1;
 

  H->hyperedges = (int **)malloc((H->M+1)*sizeof(int *));
  for(m=1;m<=H->M;m++) H->hyperedges[m] = (int *)malloc(3*sizeof(int));

  
  H->M = 0;
  for(m=1;m<=G->M;m++)
    {
      i = 1;
       
      for(j=i+1;j<=G->hyperedges[m][0];j++)
	{
	  H->M += 1;
	  H->hyperedges[H->M][0] = 2;
	  H->hyperedges[H->M][1] = G->hyperedges[m][i];
	  H->hyperedges[H->M][2] = G->hyperedges[m][j];
	}
    }
  
  
  H->hyperbonds = (int **)malloc((H->N+1)*sizeof(int *));
  H->node_rank = (int **)malloc((H->N+1)*sizeof(int *));

  create_hyperbonds_from_hyperedges (H);

}


////////////////////////////////////////
///////////////////////////////////////


void deallocate_memory_results (struct model_results *R)
{
  free(R->scores);
  free(R->tmp_scores);
  free(R->vector_error[0]);
  free(R->vector_error[1]);
  free(R->vector_error[2]);
  free(R->vector_error);
  free(R);
}

void print_results (struct model_results *R)
{
  printf("#Convergence %d %g %g\n",R->iterations,R->convergence,R->log_convergence);
  printf("#Evaluation %g %g\n",R->error, R->log_error);
  
  //int i;
  //printf("#Scores\n");
  //for(i=1;i<=R->N;i++) printf("%d %g\n",i,R->scores[i]);

  fflush(stdout);
  
}

void normalize_scores (struct model_results *R)
{
  int i;
  R->scores[0] = R->tmp_scores[0] = 0.0; 
  for(i=1;i<=R->N;i++)
    {
      R->scores[0] += log(R->scores[i]);
      R->tmp_scores[0] += log(R->tmp_scores[i]);
    }
  R->scores[0] = exp(R->scores[0]/(double)R->N);
  R->tmp_scores[0] = exp(R->tmp_scores[0]/(double)R->N);

  for(i=1;i<=R->N;i++)
    {
      R->scores[i] = R->scores[i] / R->scores[0];
      R->tmp_scores[i] = R->tmp_scores[i] / R->tmp_scores[0];
    }
  
}

void measure_convergence (struct model_results *R)
{
  double tmp;
  int i;  
  R->convergence = R->log_convergence = R->av_convergence = 0.0;
  /*
  for(i=1;i<=R->N;i++)
    {
      tmp = fabs(R->scores[i] - R->tmp_scores[i]);
      if (tmp > R->convergence) R->convergence = tmp;
      tmp = fabs(log(R->scores[i]) - log(R->tmp_scores[i]));
      if (tmp > R->log_convergence) R->log_convergence = tmp;
      tmp = fabs(R->scores[i]/(1.0+R->scores[i]) - R->tmp_scores[i]/(1.0+R->tmp_scores[i]));
      if (tmp > R->av_convergence) R->av_convergence = tmp;
    }
  */
    for(i=1;i<=R->N;i++)
    {
      tmp = R->scores[i] - R->tmp_scores[i];
      R->convergence += tmp*tmp;
      tmp = log(R->scores[i]) - log(R->tmp_scores[i]);
      R->log_convergence += tmp*tmp;
      tmp = R->scores[i]/(1.0+R->scores[i]) - R->tmp_scores[i]/(1.0+R->tmp_scores[i]);
      R->av_convergence += tmp*tmp;
    }
    R->convergence /= (double)R->N;
    R->convergence = sqrt(R->convergence);
    R->log_convergence /= (double)R->N;
    R->log_convergence = sqrt(R->log_convergence);
    R->av_convergence /= (double)R->N;
    R->av_convergence = sqrt(R->av_convergence);
}


///
void iterative_algorithm_ho_model (struct hypergraph *G, struct model_results *R, double accuracy, int max_iter)
{
  int i;
  
  R->N = G->N;
  R->iterations = 0;

  
  R->scores = (double *)malloc((R->N+1)*sizeof(double));
  R->tmp_scores = (double *)malloc((R->N+1)*sizeof(double));
  for(i=1;i<=R->N;i++)
    {
      R->scores[i] = random_number_from_logistic();
      R->tmp_scores[i] = random_number_from_logistic();
    }
  normalize_scores (R);
  

  R->vector_error = (double **)malloc(3*sizeof(double*));
  R->vector_error[0] = (double *)malloc((max_iter+1)*sizeof(double));
  R->vector_error[1] = (double *)malloc((max_iter+1)*sizeof(double));
  R->vector_error[2] = (double *)malloc((max_iter+1)*sizeof(double));
  for(i=0;i<=max_iter;i++) R->vector_error[0][i] = R->vector_error[1][i] = R->vector_error[2][i] = -1.0; 
  
 
  single_iteration_ho_model (G, R);
  normalize_scores (R);
  measure_convergence (R);


  R->vector_error[0][R->iterations] = R->log_convergence;
  R->vector_error[1][R->iterations] = R->convergence;
  R->vector_error[2][R->iterations] = R->av_convergence;

  //printf("#%d %g %g\n",R->iterations,R->log_convergence,R->convergence); fflush(stdout); 


  while (R->av_convergence>accuracy && R->iterations < max_iter)
    {
      single_iteration_ho_model (G, R);
      normalize_scores (R);
      measure_convergence (R);

      R->vector_error[0][R->iterations] = R->log_convergence;
      R->vector_error[1][R->iterations] = R->convergence;
      R->vector_error[2][R->iterations] = R->av_convergence;
      //printf("#%d %g %g\n",R->iterations,R->log_convergence,R->convergence); fflush(stdout); 
    }
  
  return;
}



void single_iteration_ho_model (struct hypergraph *G, struct model_results *R)
{
  int t, v, i, r, m, j;
  double tmp, num, den;

  R->iterations += 1;
  
  
  for(i=1;i<=R->N;i++) R->tmp_scores[i] = R->scores[i];


  for(i=1;i<=G->N;i++)
    {
      if (R->cyclic == 0) num = den = 1.0 / (R->tmp_scores[i] + 1.0);
      if (R->cyclic == 1) num = den = 1.0 / (R->scores[i] + 1.0);

      
      for(j=1;j<=G->node_rank[0][i];j++)
	{
	  r = G->node_rank[i][j];
	  m = G->hyperbonds[i][j];

	  //printf("%d %d %d\n",i,r,m); fflush(stdout);

	  if (r < G->hyperedges[m][0])
	    {
	      tmp = 0.0;
	      if (R->cyclic == 0)
                {
                  for(v=r;v<=G->hyperedges[m][0];v++) tmp += R->tmp_scores[G->hyperedges[m][v]];
                  num += (tmp - R->tmp_scores[G->hyperedges[m][r]]) / tmp;
                }
              if (R->cyclic == 1)
                {
                  for(v=r;v<=G->hyperedges[m][0];v++) tmp += R->scores[G->hyperedges[m][v]];
                  num += (tmp - R->scores[G->hyperedges[m][r]]) / tmp;
                }
	    }

	  for(t=1;t<=r-1;t++){
	    tmp = 0.0;
	    if (R->cyclic == 0) for(v=t;v<=G->hyperedges[m][0];v++) tmp += R->tmp_scores[G->hyperedges[m][v]];
            if (R->cyclic == 1) for(v=t;v<=G->hyperedges[m][0];v++) tmp += R->scores[G->hyperedges[m][v]];
	    den += 1.0 / tmp;
	  }
	  
	  
	}

      R->scores[i] = num /den;

    }
  

}










///
void iterative_algorithm_leadership_model (struct hypergraph *G, struct model_results *R, double accuracy, int max_iter)
{
  int i;
  
  R->N = G->N;
  R->iterations = 0;
  R->scores = (double *)malloc((R->N+1)*sizeof(double));
  R->tmp_scores = (double *)malloc((R->N+1)*sizeof(double));
  for(i=1;i<=R->N;i++)
    {
      R->scores[i] = random_number_from_logistic();
      R->tmp_scores[i] = random_number_from_logistic();
    }
  normalize_scores (R);

  R->vector_error = (double **)malloc(3*sizeof(double*));
  R->vector_error[0] = (double *)malloc((max_iter+1)*sizeof(double));
  R->vector_error[1] = (double *)malloc((max_iter+1)*sizeof(double));
  R->vector_error[2] = (double *)malloc((max_iter+1)*sizeof(double));
  for(i=0;i<=max_iter;i++) R->vector_error[0][i] = R->vector_error[1][i] = R->vector_error[2][i] = -1.0;



 
  single_iteration_leadership_model (G, R);
  normalize_scores (R);
  measure_convergence (R);


  R->vector_error[0][R->iterations] = R->log_convergence;
  R->vector_error[1][R->iterations] = R->convergence;
  R->vector_error[2][R->iterations] = R->av_convergence;

  
  //printf("#%d %g %g\n",R->iterations,R->log_convergence,R->convergence); fflush(stdout); 


  while (R->av_convergence>accuracy && R->iterations < max_iter)
    {
      single_iteration_leadership_model (G, R);
      normalize_scores (R);
      measure_convergence (R);

      R->vector_error[0][R->iterations] = R->log_convergence;
      R->vector_error[1][R->iterations] = R->convergence;
      R->vector_error[2][R->iterations] = R->av_convergence;
      //printf("#%d %g %g\n",R->iterations,R->log_convergence,R->convergence); fflush(stdout); 
    }
  
  return;
}


void single_iteration_leadership_model (struct hypergraph *G, struct model_results *R)
{
  int t, v, i, r, m, j;
  double tmp, num, den;

  R->iterations += 1;
  
  
  for(i=1;i<=R->N;i++) R->tmp_scores[i] = R->scores[i];


  for(i=1;i<=G->N;i++)
    {
      if (R->cyclic == 0) num = den = 1.0 / (R->tmp_scores[i] + 1.0);
      if (R->cyclic == 1) num = den = 1.0 / (R->scores[i] + 1.0);

      
      for(j=1;j<=G->node_rank[0][i];j++)
	{
	  r = G->node_rank[i][j];
	  m = G->hyperbonds[i][j];

	  //printf("%d %d %d\n",i,r,m); fflush(stdout);

	  if (r == 1)
	    {
	      tmp = 0.0;
	      if (R->cyclic == 0){
                for(v=r;v<=G->hyperedges[m][0];v++) tmp += R->tmp_scores[G->hyperedges[m][v]];
                num += (tmp - R->tmp_scores[G->hyperedges[m][r]]) / tmp;
              }
              if (R->cyclic == 1){
                for(v=r;v<=G->hyperedges[m][0];v++) tmp += R->scores[G->hyperedges[m][v]];
                num += (tmp - R->scores[G->hyperedges[m][r]]) / tmp;
              }
	    }

	  else{
	    tmp = 0.0;
	    if (R->cyclic == 0) for(v=1;v<=G->hyperedges[m][0];v++) tmp += R->tmp_scores[G->hyperedges[m][v]];
            if (R->cyclic == 1) for(v=1;v<=G->hyperedges[m][0];v++) tmp += R->scores[G->hyperedges[m][v]];
	    den += 1.0 / tmp;
	  }
	  
	  
	}

      R->scores[i] = num /den;

    }
  

}




////////////////////////////////


void create_train_test_sets (struct hypergraph *G, struct hypergraph *Gtrain, struct hypergraph *Gtest, double ratio)
{
  int i, m;
  int Mtrain = (int)((double)G->M * ratio);
  int Mtest = G->M - Mtrain;
  int *control = (int *)malloc((G->M+1)*sizeof(int));

  //////
  control[0] = 0;
  for(i=1;i<=G->M;i++) control[i] = 1;
  while(control[0]<Mtest)
    {
      i = (int)(genrand64_real3() * (double) G->M ) + 1;
      if (i >G->M) i=1;
      m = -1;
      if(control[i] == 2) m=1;
      while(m == 1)
	{
	  i += 1;
	  if (i >G->M) i=1;
	  if(control[i] ==1) m = -1;
	}
       control[0] += 1;
       control[i] = 2;
    }
  //////
  //printf(">> %d\n",control[0]);
  //for(i=1;i<=G->M;i++) printf("%d %d\n",i,control[i]);
  //printf("\n");
  //exit(0);
  
  Gtrain->N = G->N;
  Gtrain->M = Mtrain;
  Gtrain->hyperedges = (int **)malloc((Mtrain+1)*sizeof(int *));
  Gtrain->pi_values = (double *)malloc((G->N+1)*sizeof(double));
  Gtrain->hyperbonds = (int **)malloc((G->N+1)*sizeof(int *));
  Gtrain->node_rank = (int **)malloc((G->N+1)*sizeof(int *));

  Gtest->N = G->N;
  Gtest->M = Mtest;
  Gtest->hyperedges = (int **)malloc((Mtest+1)*sizeof(int *));
  Gtest->pi_values = (double *)malloc((G->N+1)*sizeof(double));
  Gtest->hyperbonds = (int **)malloc((G->N+1)*sizeof(int *));
  Gtest->node_rank = (int **)malloc((G->N+1)*sizeof(int *));

  for(i=1;i<=G->N;i++) Gtrain->pi_values[i] = Gtest->pi_values[i] = G->pi_values[i];

  //printf("-> %d %d %d\n",G->M,Gtrain->M,Gtest->M);

  Gtrain->M = Gtest->M = 0;
   for(m=1;m<=G->M;m++)
     {
       if(control[m] == 1)
	 {
	   Gtrain->M +=1;
	   //printf("#train %d %d\n",m,Gtrain->M);
	   Gtrain->hyperedges[Gtrain->M] = (int *)malloc((G->hyperedges[m][0]+1)*sizeof(int));
	   Gtrain->hyperedges[Gtrain->M][0] = G->hyperedges[m][0];
	   for(i=1;i<=G->hyperedges[m][0];i++) Gtrain->hyperedges[Gtrain->M][i] = G->hyperedges[m][i];
	 }
       if(control[m] == 2)
	 {
	   Gtest->M +=1;
	   //printf("#test %d %d\n",m,Gtest->M);
	   Gtest->hyperedges[Gtest->M] = (int *)malloc((G->hyperedges[m][0]+1)*sizeof(int));
	   Gtest->hyperedges[Gtest->M][0] = G->hyperedges[m][0];
	   for(i=1;i<=G->hyperedges[m][0];i++) Gtest->hyperedges[Gtest->M][i] = G->hyperedges[m][i];
	 }
     }


   create_hyperbonds_from_hyperedges (Gtrain);
   create_hyperbonds_from_hyperedges (Gtest);



   free(control);
}



////////////////////////
///////////////////////


///
void zermelo_iterative_algorithm_ho_model (struct hypergraph *G, struct model_results *R, double accuracy, int max_iter)
{
  int i;
  
  R->N = G->N;
  R->iterations = 0;

  
  R->scores = (double *)malloc((R->N+1)*sizeof(double));
  R->tmp_scores = (double *)malloc((R->N+1)*sizeof(double));
  for(i=1;i<=R->N;i++)
    {
      R->scores[i] = random_number_from_logistic();
      R->tmp_scores[i] = random_number_from_logistic();
    }
  normalize_scores (R);
  

  R->vector_error = (double **)malloc(3*sizeof(double*));
  R->vector_error[0] = (double *)malloc((max_iter+1)*sizeof(double));
  R->vector_error[1] = (double *)malloc((max_iter+1)*sizeof(double));
  R->vector_error[2] = (double *)malloc((max_iter+1)*sizeof(double));
  for(i=0;i<=max_iter;i++) R->vector_error[0][i] = R->vector_error[1][i] = R->vector_error[2][i] = -1.0; 
  
 
  zermelo_single_iteration_ho_model (G, R);
  normalize_scores (R);
  measure_convergence (R);


  R->vector_error[0][R->iterations] = R->log_convergence;
  R->vector_error[1][R->iterations] = R->convergence;
  R->vector_error[2][R->iterations] = R->av_convergence;

  //printf("#%d %g %g\n",R->iterations,R->log_convergence,R->convergence); fflush(stdout); 


  while (R->av_convergence>accuracy && R->iterations < max_iter)
    {
      zermelo_single_iteration_ho_model (G, R);
      normalize_scores (R);
      measure_convergence (R);

      R->vector_error[0][R->iterations] = R->log_convergence;
      R->vector_error[1][R->iterations] = R->convergence;
      R->vector_error[2][R->iterations] = R->av_convergence;
      //printf("#%d %g %g\n",R->iterations,R->log_convergence,R->convergence); fflush(stdout); 
    }
  
  return;
}


void zermelo_single_iteration_ho_model (struct hypergraph *G, struct model_results *R)
{
  int t, v, i, r, m, j;
  double tmp, num, den;

  R->iterations += 1;
  
  for(i=1;i<=R->N;i++) R->tmp_scores[i] = R->scores[i];


  for(i=1;i<=G->N;i++)
    {

      num = 1.0;
      if(R->cyclic == 0) den = 2.0 / (R->tmp_scores[i] + 1.0);
      if(R->cyclic == 1) den = 2.0 / (R->scores[i] + 1.0);

      
      for(j=1;j<=G->node_rank[0][i];j++)
	{
	  r = G->node_rank[i][j];
	  m = G->hyperbonds[i][j];

	  num += 1.0;
	  
	  for(t=1;t<=r;t++){
            tmp = 0.0;
	    if(R->cyclic == 0) for(v=t;v<=G->hyperedges[m][0];v++) tmp += R->tmp_scores[G->hyperedges[m][v]];
            if(R->cyclic == 1) for(v=t;v<=G->hyperedges[m][0];v++) tmp += R->scores[G->hyperedges[m][v]];
            den += 1.0 / tmp;
          }

	}

      R->scores[i] = num /den;

    }
  

}

