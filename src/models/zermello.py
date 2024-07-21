import random 
import os 
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models import synch_solve_equations
from src.utils.graph_tools import create_hypergraph_from_data_weight, binarize_data_weighted 

def iterate_equation_zermelo (s, scores, bond_matrix):


    #a = b = 0.0

    ##prior
    a = 1.0
    b = 2.0/(scores[s]+1.0)

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



def iterate_equation_zermelo_new (s, scores, bond_matrix):


    #a = b = 0.0

    ##prior
    a = 1.0
    b = 2.0/(scores[s]+1.0)

    for K in bond_matrix[s]:



        for r in bond_matrix[s][K]:



            a += len(bond_matrix[s][K][r])



            for t in range(0, len(bond_matrix[s][K][r])):
                for v in range(0, r+1):
                    tmp = 0.0
                    for q in range(v, K):
                        tmp += scores[bond_matrix[s][K][r][t][q]]
                    b += 1.0 / tmp


    return a/b



def compute_predicted_ratings_BT_zermello(training_set, pi_values):
    bin_data = binarize_data_weighted(training_set)
    bin_bond_matrix = create_hypergraph_from_data_weight(bin_data)

    predicted_std_scores = synch_solve_equations(bin_bond_matrix, 1000, pi_values, iterate_equation_zermelo, sens=1e-10)
    return predicted_std_scores



def compute_predicted_ratings_plackett_luce(training_set, pi_values): 
    bond_matrix = create_hypergraph_from_data_weight(training_set)
    predicted_ho_scores = synch_solve_equations(bond_matrix, 1000, pi_values, iterate_equation_zermelo_new, sens=1e-10)
 
    return predicted_ho_scores