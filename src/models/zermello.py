import random 
import os 
import sys
import numpy as np 

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from tst.tst_weight_conversion.old_newman import *

def iterate_equation_zermelo (s, scores, bond_matrix):


    #a = b = 0.0

    ##prior
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



def iterate_equation_zermelo_new (s, scores, bond_matrix):


    #a = b = 0.0

    ##prior
    a = 1.0
    b = 2.0/(scores[s]+1.0)

    if s in bond_matrix:
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
    bin_data = binarize_data_old(training_set)
    bin_bond_matrix = create_hypergraph_from_data_old(bin_data)

    predicted_std_scores, info = synch_solve_equations_old(bin_bond_matrix, 1000, pi_values, iterate_equation_zermelo, sens=1e-10)
    return predicted_std_scores, info



def compute_predicted_ratings_plackett_luce(training_set, pi_values, max_iter=1000): 
    bond_matrix = create_hypergraph_from_data_old(training_set)
    predicted_ho_scores, info = synch_solve_equations_old(bond_matrix, max_iter, pi_values, iterate_equation_zermelo_new, sens=1e-10)
 
    return predicted_ho_scores, info