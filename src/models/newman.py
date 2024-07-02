import sys
import os
import csv
import random

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.file_handlers import *
from src.utils.metrics import * 
from src.utils.graph_tools import *
from src.utils.solvers import *


def iterate_equation_newman (s, scores, bond_matrix):
    ##prior
    a = b = 1.0 / (scores[s]+1.0)
    if s in bond_matrix:

      for K in bond_matrix[s]:

          for r in bond_matrix[s][K]:

              if r < K-1:

                  for t in range(0, len(bond_matrix[s][K][r])):
                      tmp1 = tmp2 =  0.0
                      for q in range(r, K):
                          if q > r:
                              tmp1 += scores[bond_matrix[s][K][r][t][q]]
                          tmp2 += scores[bond_matrix[s][K][r][t][q]]

                      a += tmp1/tmp2

              for t in range(0, len(bond_matrix[s][K][r])):
                  for v in range(0, r):
                      tmp = 0.0
                      for q in range(v, K):
                          tmp += scores[bond_matrix[s][K][r][t][q]]
                      b += 1.0 / tmp

  #             for t in range(0, len(bond_matrix[s][K][r])):
  #                 tmp = 0.0
  #                 for q in range(0, K):
  #                     tmp += scores[bond_matrix[s][K][r][t][q]]
  #                 b += 1.0 / tmp
  #                 for q in range(0, r-1):
  #                     tmp = tmp - scores[bond_matrix[s][K][r][t][q]]
  #                     b += 1.0 / tmp

    return a/b


def compute_predicted_ratings_std(training_set, pi_values):
    bin_data = binarize_data (training_set)
    bin_bond_matrix = create_hypergraph_from_data (bin_data)

    predicted_std_scores, _ = synch_solve_equations(bin_bond_matrix, 1000, pi_values, iterate_equation_newman, sens=1e-6)

    return predicted_std_scores

def compute_predicted_ratings_std_leadership(training_set, pi_values): 
    bin_data = binarize_data_leadership (training_set)
    bin_bond_matrix = create_hypergraph_from_data (bin_data)

    predicted_std_scores, _ = synch_solve_equations(bin_bond_matrix, 1000, pi_values, iterate_equation_newman, sens=1e-6)

    return predicted_std_scores



if __name__ == '__main__':
    

    data, pi_values = read_strict_ordered_dataset('datasets/preflib_datasets/00004-00000001.soc')

    print(compute_predicted_rankings_std(data, pi_values))

