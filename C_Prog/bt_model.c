#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "bt_functions.h"
#include "mt64.h"
#include "my_sort.h"


#define eps 1e-6
#define MAX_ITER 1000


int main (int argc, char **argv)
{

  clock_t start, end;
  double cpu_time_used;


  int pid_id= time(NULL) * getpid();
  init_genrand64((unsigned)pid_id);

  //start = clock();
  ////////////////////////////

  int N, M, K1, K2, model;
  double ratio;
  N = atoi(argv[1]);
  M = atoi(argv[2]);
  K1 = atoi(argv[3]);
  K2 = atoi(argv[4]);
  model = atoi(argv[5]);
  ratio = atof(argv[6]);

  struct hypergraph *G = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct hypergraph *Gtrain = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct hypergraph *Gtest = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct model_results *R =  (struct model_results*)malloc(1 * sizeof(struct model_results));
  struct hypergraph *binG = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct hypergraph *binGtrain = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct hypergraph *binGtest = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
  struct model_results *binR =  (struct model_results*)malloc(1 * sizeof(struct model_results));
  struct model_results *leader_R =  (struct model_results*)malloc(1 * sizeof(struct model_results));

  if (model == 1)  generate_ho_model (N, M, K1, K2, G);
  else generate_leadership_model (N, M, K1, K2, G);
  compute_probability_model (G);


  create_train_test_sets (G, Gtrain, Gtest, ratio);
  compute_probability_model (Gtrain);
  compute_probability_model (Gtest);
 
  //print_hypergraph(G);


  iterative_algorithm_ho_model (Gtrain, R, eps, MAX_ITER);
  evaluate_results (Gtest, R);
  //print_results (R);


  

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
  //print_results (binR);


  iterative_algorithm_leadership_model (Gtrain, leader_R, eps, MAX_ITER);
  evaluate_results (Gtest, leader_R);


  printf("%d %d %g %g %g %g", N, M, Gtest->prior, Gtest->likelihood_ho, Gtest->likelihood_leader, binGtest->likelihood_ho);
  printf("\t");
  printf("%g %g %g %g %g %g", R->log_error, R->spearman,R->kendall, R->prior,R->likelihood_ho,R->likelihood_leader);
  printf("\t");
  printf("%g %g %g %g %g %g", leader_R->log_error, leader_R->spearman,leader_R->kendall, leader_R->prior,leader_R->likelihood_ho,leader_R->likelihood_leader);
  printf("\t");
  printf("%g %g %g %g %g %g", binR->log_error, binR->spearman,binR->kendall, binR->prior,binR->likelihood_ho,binR->likelihood_leader);
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












