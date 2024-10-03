import sys
import os
import numpy as np
from scipy.stats import logistic


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.graph_tools import create_hypergraph_from_data_weight, binarize_data_weighted, binarize_data_weighted_leadership


def normalize_scores_numpy(scores):
    scores_nonzero = scores[scores != 0]
    if len(scores_nonzero) == 0:
        return
    norm = np.exp(np.sum(np.log(scores_nonzero)) / len(scores_nonzero))
    for i in range(len(scores)):
        scores[i] /= norm

def rms_error_numpy(new_scores, old_scores):
    # Calculate the probability of beating the average player for both new and old scores
    beating_avg_new = new_scores / (new_scores + 1)
    beating_avg_old = old_scores / (old_scores + 1)
    
    # Compute the RMS error between the transformed scores
    return np.sqrt(np.mean((beating_avg_new - beating_avg_old) ** 2))


def synch_solve_equations(bond_matrix, max_iter, pi_values, method, sens=1e-6):
    players = np.array(list(pi_values.keys()))

    # scores = np.ones(len(pi_values))
    # MAP prior
    scores = np.sqrt(np.exp(logistic.rvs(size=len(pi_values))))


    normalize_scores_numpy(scores)

    # err = 1.0
    rms = 1.0
    iteration = 0
    
    while iteration < max_iter and rms > sens:
    
        tmp_scores = np.ones(len(pi_values))

        for s in range(len(scores)):
            if s in bond_matrix:
                games_with_player = bond_matrix[s]
                tmp_scores[s] = method(s, scores, games_with_player)
        
        normalize_scores_numpy(tmp_scores)
       
        # err = np.max(np.abs(np.log(tmp_scores) - np.log(scores)))
        rms = rms_error_numpy(tmp_scores, scores)

        scores = tmp_scores.copy()
        iteration += 1

        
    final_scores = {players[i]: scores[i] for i in range(len(players))}
    return final_scores, iteration



def iterate_equation_newman_weighted(player_idx, pi_values, games_with_players):
    a = b = 1.0 / (pi_values[player_idx] + 1.0)
    tolerance = 1e-10

    for K, position, game, weight in games_with_players:
        score_sums = [pi_values[p] for p in game]
        cumulative_sum = [0] * (K + 1)

        # Calculate cumulative sums
        for j in range(1, K + 1):
            cumulative_sum[j] = cumulative_sum[j - 1] + score_sums[j - 1]

        if position < K - 1:
            tmp1 = cumulative_sum[K] - cumulative_sum[position+1]
            tmp2 = tmp1 + score_sums[position]
            if tmp2 != 0:
                a += weight * (tmp1 / tmp2)

        for v in range(position):
            tmp = cumulative_sum[K] - cumulative_sum[v]
            if tmp != 0 :
                b += weight * (1.0 / tmp)

    return a / b

def iterate_equation_newman_leadership_weighted(player_idx, pi_values, games_with_players):
    a = b = 1.0 / (pi_values[player_idx] + 1.0)
    tolerance = 1e-10

    for K, position, game, weight in games_with_players:
        
        score_sums = [pi_values[game[p]] for p in range(K)]
        cumulative_sum = [0] * (K + 1)
        for j in range(1, K + 1):
            cumulative_sum[j] = cumulative_sum[j - 1] + score_sums[j - 1]

        if position == 0:
            tmp1 = cumulative_sum[K] - cumulative_sum[position + 1]
            tmp2 = tmp1 + score_sums[position]
            if tmp2 != 0 :
                a += weight * (tmp1 / tmp2)
        else:
            tmp = cumulative_sum[K]
            if tmp != 0:
                b += weight * (1.0 / tmp)
    

    return a/b




def compute_predicted_ratings_BT(training_set, pi_values):
    bin_data = binarize_data_weighted(training_set)
    bin_bond_matrix = create_hypergraph_from_data_weight(bin_data)

    predicted_std_scores, iter = synch_solve_equations(bin_bond_matrix, 10000, pi_values, iterate_equation_newman_weighted, sens=1e-6)
    return predicted_std_scores

def compute_predicted_ratings_BT_leadership(training_set, pi_values): 
    bin_data = binarize_data_weighted_leadership(training_set)
    bin_bond_matrix = create_hypergraph_from_data_weight(bin_data)

    predicted_std_scores, iter = synch_solve_equations(bin_bond_matrix, 10000, pi_values, iterate_equation_newman_leadership_weighted, sens=1e-6)
    return predicted_std_scores

def compute_predicted_ratings_HO_BT(training_set, pi_values): 
    bond_matrix = create_hypergraph_from_data_weight(training_set)
    predicted_ho_scores, iter = synch_solve_equations(bond_matrix, 10000, pi_values, iterate_equation_newman_weighted, sens=1e-6)
    return predicted_ho_scores


def compute_predicted_ratings_HOL_BT(training_set, pi_values):
    bond_matrix = create_hypergraph_from_data_weight(training_set)
    predicted_hol_scores, iter = synch_solve_equations (bond_matrix, 10000, pi_values, iterate_equation_newman_leadership_weighted, sens=1e-6)
    return predicted_hol_scores

