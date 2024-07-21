import os
import sys
import csv
import random
import numpy as np
from numba import njit, prange, types
from numba.typed import List

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import *

@njit
def normalize_scores(scores):
    norm = np.exp(np.sum(np.log(scores)) / len(scores))
    scores /= norm



def synch_solve_equations(bond_matrix, max_iter, pi_values, method, sens=1e-10):
    players = np.array(list(pi_values.keys()))
    pi_values_array = np.array([pi_values[player] for player in players])

    scores = np.ones(len(pi_values_array))
    normalize_scores(scores)
   
    err = 1.0
    iteration = 0
    
    while iteration < max_iter and err > sens:
        err = 0.0
        tmp_scores = np.zeros_like(scores)
        for s in range(len(scores)):
            if s in bond_matrix:
                games_with_player = bond_matrix[s]
                tmp_scores[s] = method(s, scores, games_with_player)
        
        normalize_scores(tmp_scores)
        
        err = np.max(np.abs(tmp_scores - scores))
        scores = tmp_scores.copy()
        
        iteration += 1

    final_scores = {players[i]: scores[i] for i in range(len(players))}
    return final_scores, iteration


def iterate_equation_newman_weighted(player_idx, pi_values, games_with_players):
    a = b = 1.0 / (pi_values[player_idx] + 1.0)

    for i in range(len(games_with_players)):

        K, position, game, weight = games_with_players[i]
        
        score_sums = np.array([pi_values[game[p]] for p in range(K)])
        if position < K - 1:
            tmp1 = np.sum(score_sums[position+1:K])
            tmp2 = np.sum(score_sums[position:K])
            if tmp2 != 0:
                a += weight * (tmp1 / tmp2)
        for v in range(position):
            tmp = np.sum(score_sums[v:K])
            if tmp != 0:
                b += weight * (1.0 / tmp)


    return a / b


def iterate_equation_newman_leadership_weighted(player_idx, pi_values, games_with_players):
    a = b = 1.0 / (pi_values[player_idx] + 1.0)

    for i in range(len(games_with_players)):

        K, position, game, weight = games_with_players[i]
        
        score_sums = np.array([pi_values[game[p]] for p in range(K)])
        if position == 0:
            tmp1 = np.sum(score_sums[1:K])
            tmp2 = np.sum(score_sums[0:K])
            if tmp2 != 0:
                a += weight * (tmp1 / tmp2)
        else:
            tmp = np.sum(score_sums[0:K])
            if tmp != 0:
                b += weight * (1.0 / tmp)

    return a/b



def numpy_std(training_set, pi_values):
    bin_data = binarize_data_weighted(training_set)
    bin_bond_matrix = create_hypergraph_from_data_weight(bin_data)

    predicted_std_scores, _ = synch_solve_equations(bin_bond_matrix, 1000, pi_values, iterate_equation_newman_weighted, sens=1e-10)

    return predicted_std_scores

def numpy_std_leadership(training_set, pi_values): 
    bin_data = binarize_data_weighted_leadership(training_set)
    bin_bond_matrix = create_hypergraph_from_data_weight(bin_data)

    predicted_std_scores, _ = synch_solve_equations(bin_bond_matrix, 1000, pi_values, iterate_equation_newman_weighted, sens=1e-6)

    return predicted_std_scores

def numpy_ho(training_set, pi_values): 
    bond_matrix = create_hypergraph_from_data_weight(training_set)
    predicted_ho_scores, _ = synch_solve_equations(bond_matrix, 1000, pi_values, iterate_equation_newman_weighted, sens=1e-10)

    return predicted_ho_scores


def numpy_hol(training_set, pi_values):
    bond_matrix = create_hypergraph_from_data_weight(training_set)
    predicted_hol_scores, _ = synch_solve_equations (bond_matrix, 1000, pi_values, iterate_equation_newman_leadership_weighted, sens=1e-10)

    return predicted_hol_scores