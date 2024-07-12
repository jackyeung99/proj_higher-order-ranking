import sys
import os
import csv
import random
import concurrent.futures
import numpy as np
from numba import njit,jit, prange, types
from numba.typed import List
import math

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import *
from src.models.cython.cpython_newman import iterate_equation_newman_weighted, iterate_equation_newman_leadership_weighted, normalize_scores, synch_solve_equations


# def normalize_scores(scores):
#     scores_nonzero = scores[scores != 0]
#     if len(scores_nonzero) == 0:
#         return
#     norm = np.exp(np.sum(np.log(scores_nonzero)) / len(scores_nonzero))
#     for i in range(len(scores)):
#         scores[i] /= norm
        

# def synch_solve_equations(bond_matrix, max_iter, pi_values, method, sens=1e-10):
#     players = np.array(list(pi_values.keys()))
#     scores = np.ones(len(pi_values))
#     normalize_scores(scores)
   
#     err = 1.0
#     iteration = 0
    
#     while iteration < max_iter and err > sens:
#         err = 0
#         tmp_scores = np.zeros(len(scores))

#         for s in range(len(scores)):
#             if s in bond_matrix:
#                 games_with_player = bond_matrix[s]
#                 tmp_scores[s] = method(s, scores, games_with_player)
        
#         normalize_scores(tmp_scores)
       
#         err = np.max(np.abs(tmp_scores - scores))
#         scores = tmp_scores.copy()
        
#         iteration += 1
        
#     final_scores = {players[i]: scores[i] for i in range(len(players))}
#     return final_scores


# def iterate_equation_newman_weighted(player_idx, pi_values, games_with_players):
#     a = b = 1.0 / (pi_values[player_idx] + 1.0)

#     for K, position, game, weight in games_with_players:
#         score_sums = [pi_values[p] for p in game]
#         cumulative_sum = [0] * (K + 1)

#         # Calculate cumulative sums
#         for j in range(1, K + 1):
#             cumulative_sum[j] = cumulative_sum[j - 1] + score_sums[j - 1]

#         if position < K - 1:
#             tmp1 = cumulative_sum[K] - cumulative_sum[position + 1]
#             tmp2 = tmp1 + score_sums[position]
#             if tmp2 != 0:
#                 a += weight * (tmp1 / tmp2)

#         for v in range(position):
#             tmp = cumulative_sum[K] - cumulative_sum[v]
#             if tmp != 0:
#                 b += weight * (1.0 / tmp)

#     return a / b

# def iterate_equation_newman_leadership_weighted(player_idx, pi_values, games_with_players):
#     a = b = 1.0 / (pi_values[player_idx] + 1.0)

#     for K, position, game, weight in games_with_players:
        
#         score_sums = [pi_values[game[p]] for p in range(K)]
#         cumulative_sum = [0] * (K + 1)
#         for j in range(1, K + 1):
#             cumulative_sum[j] = cumulative_sum[j - 1] + score_sums[j - 1]

#         if position == 0:
#             tmp1 = cumulative_sum[K] - cumulative_sum[position + 1]
#             tmp2 = tmp1 + score_sums[position]
#             if tmp2 != 0:
#                 a += weight * (tmp1 / tmp2)
#         else:
#             tmp = cumulative_sum[K]
#             if tmp != 0:
#                 b += weight * (1.0 / tmp)
    

#     return a/b



def compute_predicted_ratings_std(training_set, pi_values):
    bin_data = binarize_data_weighted(training_set)
    bin_bond_matrix = create_hypergraph_from_data_weight(bin_data)

    predicted_std_scores = synch_solve_equations(bin_bond_matrix, 1000, pi_values, iterate_equation_newman_weighted, sens=1e-10)
    return predicted_std_scores

def compute_predicted_ratings_std_leadership(training_set, pi_values): 
    bin_data = binarize_data_weighted_leadership(training_set)
    bin_bond_matrix = create_hypergraph_from_data_weight(bin_data)

    predicted_std_scores = synch_solve_equations(bin_bond_matrix, 1000, pi_values, iterate_equation_newman_leadership_weighted, sens=1e-10)

    return predicted_std_scores

def compute_predicted_ratings_ho(training_set, pi_values): 
    bond_matrix = create_hypergraph_from_data_weight(training_set)
    predicted_ho_scores = synch_solve_equations(bond_matrix, 1000, pi_values, iterate_equation_newman_weighted, sens=1e-10)
 
    return predicted_ho_scores


def compute_predicted_ratings_hol(training_set, pi_values):
    bond_matrix = create_hypergraph_from_data_weight(training_set)
    predicted_hol_scores = synch_solve_equations (bond_matrix, 1000, pi_values, iterate_equation_newman_leadership_weighted, sens=1e-10)

    return predicted_hol_scores

