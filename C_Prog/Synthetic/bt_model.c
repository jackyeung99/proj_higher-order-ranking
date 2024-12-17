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
#define CYCLIC 1



int main (int argc, char **argv)
{

  clock_t start, end;
  double cpu_time_used;

  int pid_id= time(NULL) * getpid();
  init_genrand64((unsigned)pid_id);

  //start = clock();
  ////////////////////////////

  int model;
  double ratio;
  char filename_idx[100];
  char filename_data[100];
  sprintf(filename_idx,"%s",argv[1]);
  sprintf(filename_data,"%s",argv[2]);
  ratio = atof(argv[3]);
  model = atof(argv[4]);

  struct hypergraph *G = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct hypergraph *Gtrain = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct hypergraph *Gtest = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct model_results *R =  (struct model_results*)malloc(1 * sizeof(struct model_results));
  struct hypergraph *binG = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct hypergraph *binGtrain = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct hypergraph *binGtest = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct model_results *binR =  (struct model_results*)malloc(1 * sizeof(struct model_results));
  struct model_results *leader_R =  (struct model_results*)malloc(1 * sizeof(struct model_results));

  //
  R->cyclic = CYCLIC;
  binR->cyclic = CYCLIC;
  leader_R->cyclic = CYCLIC;
  //

  
  // if (model == 1)  generate_ho_model (N, M, K1, K2, G);
  // else generate_leadership_model (N, M, K1, K2, G);


  read_index_file (filename_idx, G);
  read_data_file (filename_data, G);

  compute_probability_model (G);
  // printf("Number of Test Edges:%d\n", G->M);

  create_train_test_sets (G, Gtrain, Gtest, ratio);
  compute_probability_model (Gtrain);
  compute_probability_model (Gtest);
  // printf("Number of Train Edges:%d\n", Gtrain->M);
  // printf("Number of Test Edges:%d\n", Gtest->M);
 
  // print_hypergraph(G);


  iterative_algorithm_ho_model (Gtrain, R, eps, MAX_ITER);
  evaluate_results (Gtest, R);
  // print_results (R);


  

  if(model == 1)
    {
      binarize_ho_model (G, binG);
      binarize_ho_model (Gtrain, binGtrain);
      binarize_ho_model (Gtest, binGtest);
    }
  else
    {
      binarize_leadership_model (G, binG);
      binarize_leadership_model (Gtrain, binGtrain);
      binarize_leadership_model (Gtest, binGtest);
    }
  compute_probability_model (binG);
  compute_probability_model (binGtrain);
  compute_probability_model (binGtest);



  iterative_algorithm_ho_model (binGtrain, binR, eps, MAX_ITER);
  evaluate_results (Gtest, binR);
  // print_results (binR);


  iterative_algorithm_leadership_model (Gtrain, leader_R, eps, MAX_ITER);
  evaluate_results (Gtest, leader_R);


  printf("%d %d %g %g %g %g",G->N, G->M, Gtest->prior, Gtest->likelihood_ho, Gtest->likelihood_leader, binGtest->likelihood_ho);
  printf(";;;");
  printf("%g %g %g %g %g %g", R->av_error, R->spearman,R->kendall, R->prior, R->likelihood_ho, R->likelihood_leader);
  printf(";;;");
  printf("%g %g %g %g %g %g", leader_R->av_error, leader_R->spearman,leader_R->kendall, leader_R->prior,leader_R->likelihood_ho,leader_R->likelihood_leader);
  printf(";;;");
  printf("%g %g %g %g %g %g", binR->av_error, binR->spearman, binR->kendall, binR->prior,binR->likelihood_ho, binR->likelihood_leader);

  printf(";;;");
  printf("%d %d %d\n",R->iterations, leader_R->iterations,binR->iterations);

  printf("\n");
  
  /*
  int i;
  for(i=1;i<=G->N;i++)
    {
      printf("%d %g %g %g\n",i,G->pi_values[i],R->scores[i],binR->scores[i]);
    }
  */
  
  ////////////////////////////
  //end = clock();
  //cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  //printf("#Total time : %g\n",cpu_time_used); fflush(stdout);



  deallocate_memory (G);
  deallocate_memory (Gtrain);
  deallocate_memory (Gtest);
  deallocate_memory_results (R);
  deallocate_memory (binG);
  deallocate_memory (binGtrain);
  deallocate_memory (binGtest);
  deallocate_memory_results (binR);
  deallocate_memory_results (leader_R);
 			
  
  return 0;

}












