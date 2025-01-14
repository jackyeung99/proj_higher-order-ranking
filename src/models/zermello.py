import random 
import os 
import sys
import numpy as np 

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import *
from src.models.BradleyTerry import synch_solve_equations



def iterate_equation_zermelo (s, scores, bond_matrix):


    a = 1.0
    b = 2.0/(scores[s]+1.0)
   
    if s in bond_matrix:

        for K in bond_matrix[s]:



            for r in bond_matrix[s][K]:


                if r < K-1:
                    a += len(bond_matrix[s][K][r])

                    for t in range(0, len(bond_matrix[s][K][r])):
                        tmp = 0.0
                        for q in range(r, K):
                            tmp += scores[bond_matrix[s][K][r][t][q]]
                        b += 1.0 / tmp


                for t in range(0, len(bond_matrix[s][K][r])):
                    for v in range(0, r):
                        tmp = 0.0
                        for q in range(v, K):
                            tmp += scores[bond_matrix[s][K][r][t][q]]
                        b += 1.0 / tmp
#                     print ('> ', tmp)


#             for t in range(0, len(bond_matrix[s][K][r])):
#                 tmp = 0.0
#                 for q in range(0, K):
#                     tmp += scores[bond_matrix[s][K][r][t][q]]
#                 b += 1.0 / tmp
#                 print ('>> ', tmp)
#                 for q in range(0, r-1):
#                     tmp = tmp - scores[bond_matrix[s][K][r][t][q]]
#                     print ('>> ', tmp)
#                     b += 1.0 / tmp


    return a/b   



# LEADERSHIP variant 
def iterate_equation_zermelo_new (s, scores, hypergraph):

    
    if s in hypergraph:
        a = 1.0
        b = 2.0/(scores[s]+1.0)

        for K in hypergraph[s]:
            for r in hypergraph[s][K]:
                a += len(hypergraph[s][K][r])

                for t in range(0, len(hypergraph[s][K][r])):
                    for v in range(0, r+1):
                        tmp = 0.0
                        for q in range(v, K):
                            tmp += scores[hypergraph[s][K][r][t][q]]
                        b += 1.0 / tmp

        return a/b   





def compute_predicted_ratings_BT_zermello(training_set, pi_values, verbose=False):
    bin_data = binarize_data(training_set)
    hyper_graph = create_hypergraph_from_data(bin_data)
    predicted_scores, info = synch_solve_equations(hyper_graph, pi_values, iterate_equation_zermelo)

    if verbose:
        return predicted_scores, info
    else:
        return predicted_scores




def compute_predicted_ratings_plackett_luce(training_set, pi_values, verbose=False): 
    hyper_graph = create_hypergraph_from_data(training_set)
    predicted_scores, info = synch_solve_equations(hyper_graph, pi_values, iterate_equation_zermelo)
    
    if verbose:
        return predicted_scores, info
    else:
        return predicted_scores
