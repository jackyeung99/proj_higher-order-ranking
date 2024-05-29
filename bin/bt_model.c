#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "bt_functions.h"
#include "mt64.h"




int main (int argc, char **argv)
{

	clock_t start, end;
	double cpu_time_used;


	int pid_id= time(NULL) * getpid();
	init_genrand64((unsigned)pid_id);

	//start = clock();
	////////////////////////////

	int N, M, K1, K2, MODEL;
	N = atoi(argv[1]);
	M = atoi(argv[2]);
	K1 = atoi(argv[3]);
	K2 = atoi(argv[4]);
    MODEL = atoi(argv[5]);

    if (MODEL == 0) {
        printf("Ground truth sampled from H.O. model");
    }
    else {
        printf("Ground truth sampled from H.O.L. model");
    }
    printf("\n");

	struct hypergraph *G = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
	struct model_results *R =  (struct model_results*)malloc(1 * sizeof(struct model_results));
	struct hypergraph *binG = (struct hypergraph*)malloc(1 * sizeof(struct hypergraph));
	struct model_results *binR =  (struct model_results*)malloc(1 * sizeof(struct model_results));
	struct model_results *leader_R =  (struct model_results*)malloc(1 * sizeof(struct model_results));


    if (MODEL == 0) {
        generate_ho_model (N, M, K1, K2, G);
    }
    else {
        generate_leadership_model (N, M, K1, K2, G);
    }

	compute_probability_model (G);

 
	//print_hypergraph(G);


	iterative_algorithm_ho_model (G, R, 1.0/(double)N, N);

	evaluate_results (G, R);
	//print_results (R);


	

    if (MODEL == 0) {
        binarize_ho_model (G, binG);
    }
    else {
        binarize_leadership_model (G, binG);
    }
	compute_probability_model (binG);


	
	iterative_algorithm_ho_model (binG, binR, 1.0/(double)N, N);

	evaluate_results (G, binR);
	//print_results (binR);


	iterative_algorithm_leadership_model (G, leader_R, 1.0/(double)N, N);
	evaluate_results (G, leader_R);

	printf("RMS's \t Dyad: %g  HO: %g  HOL: %g",
	     binR->log_error,
	     R->log_error,
	     leader_R->log_error
	);

    const char filename_HO[] = "scores_HO.txt";
    const char filename_HOL[] = "scores_HOL.txt";
    const char filename_bin[] = "scores_bin.txt";
    print_scores_to_file(R, filename_HO);
    print_scores_to_file(leader_R, filename_HOL);
    print_scores_to_file(binR, filename_bin);
    
	// printf("%d %d %g %g %g",
	//      N,
	//      M,
	//      R->log_error,
	//      binR->log_error,
	//      leader_R->log_error
	// );
	printf("\n");

	// printf("%g %g %g %g %g %g %g %g %g",
 //        R->likelihood_ho,
 //        R->likelihood_leader,
 //        R->prior,
 //        binR->likelihood_ho,
 //        binR->likelihood_leader,
 //        binR->prior,
 //        leader_R->likelihood_ho,
 //        leader_R->likelihood_leader,
 //        leader_R->prior
 //    );
 //    printf("\n");

	// printf("%g %g %g %g %g %g",
 //        G->likelihood_ho,
 //        G->likelihood_leader,
 //        G->prior,
 //        binG->likelihood_ho,
 //        binG->likelihood_leader,
 //        binG->prior
 //    );
	// printf("\n");
	
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
	deallocate_memory_results (R);
	deallocate_memory (binG);
	deallocate_memory_results (binR);
	deallocate_memory_results (leader_R);
 			
	
	return 0;
}

