#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "bt_functions.h"
#include "mt64.h"
#include "my_sort.h"


#define eps 1e-6
#define MAX_ITER 10000
#define CYCLIC 1


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
  double ratio;
  int model;
  sprintf(filename_idx,"%s",argv[1]);
  sprintf(filename_data,"%s",argv[2]);
  model =atoi(argv[3]);
  ratio = atof(argv[4]);
  
  
  //printf("# %s\n",filename_idx);
  //printf("# %s\n",filename_data);
  //printf("# %d\n",model);
  //printf("# %g\n",ratio);

  char **names;
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

  //////////////
  read_index_file (filename_idx, G, names);
  read_data_file (filename_data, G);
  compute_probability_model (G);


  
  //print_hypergraph(G);
  

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


  printf("%s %s %g %g %g %g", filename_idx, filename_data, Gtest->prior, Gtest->likelihood_ho, Gtest->likelihood_leader, binGtest->likelihood_ho);
  printf(";;;");
  printf("%g %g %g %g %g %g", R->av_error, R->spearman,R->kendall, R->prior,R->likelihood_ho,R->likelihood_leader);
  printf(";;;");
  printf("%g %g %g %g %g %g", leader_R->av_error, leader_R->spearman,leader_R->kendall, leader_R->prior,leader_R->likelihood_ho,leader_R->likelihood_leader);
  printf(";;;");
  printf("%g %g %g %g %g %g", binR->av_error, binR->spearman,binR->kendall, binR->prior,binR->likelihood_ho,binR->likelihood_leader);
  printf(";;;");
  printf("%d %d %d\n",R->iterations, leader_R->iterations,binR->iterations);
  printf("\n");
  
  
  //for(i=1;i<=G->N;i++)
  //  {
  //    printf("%d %g\n",i,R->scores[i]);
  //  }
  
  
  ////////////////////////////
  //end = clock();
  //cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  //printf("#Total time : %g\n",cpu_time_used); fflush(stdout);


  //for (i=1;i<=G->N;i++) free(names[i]);
  //free(names);
  
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












