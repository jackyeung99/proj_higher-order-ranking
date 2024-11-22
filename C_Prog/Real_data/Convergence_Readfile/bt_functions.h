struct hypergraph{
  int N;
  int M;
  int **hyperedges;
  int **hyperbonds;
  int **node_rank;
  double *pi_values;
  double likelihood_ho;
  double likelihood_leader;
  double prior;
};


struct model_results
{
  int N;
  int iterations;
  double convergence;
  double log_convergence;
  double av_convergence;
  double likelihood_ho;
  double likelihood_leader;
  double prior;
  double error;
  double log_error;
  double av_error;
  double spearman;
  double kendall;
  double *scores;
  double *tmp_scores;
  double **vector_error;
  int cyclic;
};


void read_index_file (char *filename, struct hypergraph *G, char **names);
void read_data_file (char *filename, struct hypergraph *G);

void deallocate_memory (struct hypergraph *G);
void print_hypergraph(struct hypergraph *G);
double random_number_from_logistic (void);
void normalize_pi_values (struct hypergraph *G);

void create_hyperbonds_from_hyperedges (struct hypergraph* G);


void generate_pi_values(struct hypergraph *G);
void generate_ho_model (int N, int M, int K1, int K2, struct hypergraph *G);
void create_hyperedge_ho_model (struct hypergraph *G, int m, int **control);
void compute_probability_model (struct hypergraph *G);

void generate_leadership_model (int N, int M, int K1, int K2, struct hypergraph *G);
void create_hyperedge_leadership_model (struct hypergraph *G, int m, int **control);


void normalize_scores (struct model_results *R);
void deallocate_memory_results (struct model_results *R);
void print_results (struct model_results *R);

void iterative_algorithm_ho_model (struct hypergraph *G, struct model_results *R, double accuracy, int max_iter);
void single_iteration_ho_model (struct hypergraph *G, struct model_results *R);

void zermelo_iterative_algorithm_ho_model (struct hypergraph *G, struct model_results *R, double accuracy, int max_iter);
void zermelo_single_iteration_ho_model (struct hypergraph *G, struct model_results *R);

void single_iteration_leadership_model (struct hypergraph *G, struct model_results *R);
void iterative_algorithm_leadership_model (struct hypergraph *G, struct model_results *R, double accuracy, int max_iter);

void evaluate_results (struct hypergraph *G, struct model_results *R);
double spearman_correlation (int N, double *vec1, double *vec2);
double kendall_correlation (int N, double *vec1, double *vec2);



void binarize_ho_model (struct hypergraph *G, struct  hypergraph *H);
void binarize_leadership_model (struct hypergraph *G, struct  hypergraph *H);


void create_train_test_sets (struct hypergraph *G, struct hypergraph *Gtrain, struct hypergraph *Gtest, double ratio);
