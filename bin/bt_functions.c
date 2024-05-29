#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "bt_functions.h"
#include "mt64.h"


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

  //prior
  G->prior = 0.0;
  for(i=1;i<=G->N;i++) G->prior += log(G->pi_values[i]) - 2.0 * log(1.0 + G->pi_values[i]);
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
  R->error = R->log_error = 0.0;
  
  for(i=1;i<=G->N;i++)
    {
      R->error += (G->pi_values[i]-R->scores[i])*(G->pi_values[i]-R->scores[i]);
      R->log_error += (log(G->pi_values[i])-log(R->scores[i]))*(log(G->pi_values[i])-log(R->scores[i]));
    }

  R->error = R->error / (double)R->N;
  R->error = sqrt(R->error);
  R->log_error = R->log_error / (double)R->N;
  R->log_error = sqrt(R->log_error);



  //likelihood
  R->likelihood_ho = R->likelihood_leader = 0.0;
  for(m=1;m<=G->M;m++)
    {

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

  //prior
  R->prior = 0.0;
  for(i=1;i<=G->N;i++) R->prior += log(R->scores[i]) - 2.0 * log(1.0 + R->scores[i]);

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
  R->convergence = R->log_convergence = 0.0;
  for(i=1;i<=R->N;i++)
    {
      tmp = fabs(R->scores[i] - R->tmp_scores[i]);
      if (tmp > R->convergence) R->convergence = tmp;
      tmp = fabs(log(R->scores[i]) - log(R->tmp_scores[i]));
      if (tmp > R->log_convergence) R->log_convergence = tmp;
    }
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


 
  single_iteration_ho_model (G, R);
  normalize_scores (R);
  measure_convergence (R);

  //printf("#%d %g %g\n",R->iterations,R->log_convergence,R->convergence); fflush(stdout); 


  while (R->log_convergence>accuracy && R->iterations < max_iter)
    {
      single_iteration_ho_model (G, R);
      normalize_scores (R);
      measure_convergence (R);
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
      num = den = 1.0 / (R->tmp_scores[i] + 1.0);
      
      for(j=1;j<=G->node_rank[0][i];j++)
	{
	  r = G->node_rank[i][j];
	  m = G->hyperbonds[i][j];

	  //printf("%d %d %d\n",i,r,m); fflush(stdout);

	  if (r < G->hyperedges[m][0])
	    {
	      tmp = 0.0;
	      for(v=r;v<=G->hyperedges[m][0];v++) tmp += R->tmp_scores[G->hyperedges[m][v]];
	      num += (tmp - R->tmp_scores[G->hyperedges[m][r]]) / tmp;
	    }

	  for(t=1;t<=r-1;t++){
	    tmp = 0.0;
	    for(v=t;v<=G->hyperedges[m][0];v++) tmp += R->tmp_scores[G->hyperedges[m][v]];
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


 
  single_iteration_leadership_model (G, R);
  normalize_scores (R);
  measure_convergence (R);

  //printf("#%d %g %g\n",R->iterations,R->log_convergence,R->convergence); fflush(stdout); 


  while (R->log_convergence>accuracy && R->iterations < max_iter)
    {
      single_iteration_leadership_model (G, R);
      normalize_scores (R);
      measure_convergence (R);
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
      num = den = 1.0 / (R->tmp_scores[i] + 1.0);
      
      for(j=1;j<=G->node_rank[0][i];j++)
	{
	  r = G->node_rank[i][j];
	  m = G->hyperbonds[i][j];

	  //printf("%d %d %d\n",i,r,m); fflush(stdout);

	  if (r == 1)
	    {
	      tmp = 0.0;
	      for(v=r;v<=G->hyperedges[m][0];v++) tmp += R->tmp_scores[G->hyperedges[m][v]];
	      num += (tmp - R->tmp_scores[G->hyperedges[m][r]]) / tmp;
	    }

	  else{
	    tmp = 0.0;
	    for(v=1;v<=G->hyperedges[m][0];v++) tmp += R->tmp_scores[G->hyperedges[m][v]];
	    den += 1.0 / tmp;
	  }
	  
	  
	}

      R->scores[i] = num /den;

    }
  

}



/* Added - DK */
void print_scores(struct model_results *R)
{
  int i;
  for(i=1;i<=R->N;i++) {
        printf("%d %g\n",i,R->scores[i]);
    }

    fflush(stdout);
}

void print_scores_to_file(struct model_results *R, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error while opening file to dump scores!");
        exit(1);
    }

    int i;
    for(i=1;i<=R->N;i++) {
        fprintf(file, "%d %g\n",i,R->scores[i]);
    }

    fclose(file);
}

void print_true_scores_to_file(struct hypergraph *G, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error while opening file to dump scores!");
        exit(1);
    }

    int i;
    for(i=1; i<=G->N; i++) {
        fprintf(file, "%d %g\n", i, G->pi_values[i]);
    }

    fclose(file);
}
