#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "bt_functions.h"
#include "mt64.h"
#include "my_sort.h"


#define eps 1e-6
#define MAX_ITER 100000



int main (int argc, char **argv)
{
  int i;
  clock_t start, end;
  double cpu_time_used;


  int pid_id= time(NULL) * getpid();
  init_genrand64((unsigned)pid_id);

  //start = clock();
  ////////////////////////////

  char filename_idx[100];
  char filename_data[100];
  sprintf(filename_idx,"%s",argv[1]);
  sprintf(filename_data,"%s",argv[2]);

  
  
  //printf("# %s\n",filename_idx);
  //printf("# %s\n",filename_data);
  //printf("# %d\n",model);
  //printf("# %g\n",ratio);

  char **names;
  struct hypergraph *G = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct hypergraph *binG = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct model_results *R =  (struct model_results*)malloc(1 * sizeof(struct model_results));
  struct model_results *RL =  (struct model_results*)malloc(1 * sizeof(struct model_results));
  struct model_results *binR =  (struct model_results*)malloc(1 * sizeof(struct model_results));
  struct model_results *binRL =  (struct model_results*)malloc(1 * sizeof(struct model_results));




  //////////////
  read_index_file (filename_idx, G, names);
  read_data_file (filename_data, G);
  compute_probability_model (G);


  //initialize to get same initial condition
  R->N = G->N;
  R->scores = (double *)malloc((R->N+1)*sizeof(double));
  R->tmp_scores = (double *)malloc((R->N+1)*sizeof(double));
  for(i=1;i<=R->N;i++)
    {								  
      R->scores[i] = random_number_from_logistic();
      R->tmp_scores[i] = random_number_from_logistic();
    }                                                                                                                   
  normalize_scores (R);                                                                                                 

  RL->N = G->N;
  RL->scores = (double *)malloc((RL->N+1)*sizeof(double));
  RL->tmp_scores = (double *)malloc((RL->N+1)*sizeof(double));
  for(i=1;i<=RL->N;i++)
    {								  
      RL->scores[i] = R->scores[i];
      RL->tmp_scores[i] = R->tmp_scores[i];
    }                                                                                                                   
  normalize_scores (RL);


  binR->N = G->N;
  binR->scores = (double *)malloc((binR->N+1)*sizeof(double));
  binR->tmp_scores = (double *)malloc((binR->N+1)*sizeof(double));
  for(i=1;i<=binR->N;i++)
    {								  
      binR->scores[i] = R->scores[i];
      binR->tmp_scores[i] = R->tmp_scores[i];
    }                                                                                                                   
  normalize_scores (binR);

  binRL->N = G->N;
  binRL->scores = (double *)malloc((binRL->N+1)*sizeof(double));
  binRL->tmp_scores = (double *)malloc((binRL->N+1)*sizeof(double));
  for(i=1;i<=binRL->N;i++)
    {								  
      binRL->scores[i] = R->scores[i];
      binRL->tmp_scores[i] = R->tmp_scores[i];
    }                                                                                                                   
  normalize_scores (binRL);                                                                                                

  ////////////////////////////////

  
 

  iterative_algorithm_ho_model (G, R, eps, MAX_ITER);
  zermelo_iterative_algorithm_ho_model (G, RL, eps, MAX_ITER);

  binarize_ho_model (G, binG);
  iterative_algorithm_ho_model (binG, binR, eps, MAX_ITER);
  zermelo_iterative_algorithm_ho_model (binG, binRL, eps, MAX_ITER);


  for(i=0;i<=MAX_ITER;i++) if(R->vector_error[0][i]>0) printf("%d %g %g %g\t",i,R->vector_error[0][i],R->vector_error[1][i],R->vector_error[2][i]);
  printf(";;;");

  for(i=0;i<=MAX_ITER;i++) if(RL->vector_error[0][i]>0) printf("%d %g %g %g\t",i,RL->vector_error[0][i],RL->vector_error[1][i],RL->vector_error[2][i]);
  printf(";;;");

  for(i=0;i<=MAX_ITER;i++) if(binR->vector_error[0][i]>0) printf("%d %g %g %g\t",i,binR->vector_error[0][i],binR->vector_error[1][i],binR->vector_error[2][i]);
  printf(";;;");

    for(i=0;i<=MAX_ITER;i++) if(binRL->vector_error[0][i]>0) printf("%d %g %g %g\t",i,binRL->vector_error[0][i],binRL->vector_error[1][i],binRL->vector_error[2][i]);
  printf("\n");


  //for(i=1;i<=G->N;i++) printf("%d %g %g\n",i,R->scores[i],RL->scores[i]);

  
  
  deallocate_memory (G);
  deallocate_memory_results (R);
  deallocate_memory_results (RL);
  deallocate_memory (binG);
  deallocate_memory_results (binR);
  deallocate_memory_results (binRL);
  
 			
  
  return 0;

}












