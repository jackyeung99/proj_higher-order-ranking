import sys
import os 
import pytest
import cProfile
import pstats

import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models.BradleyTerry import *
from src.utils.graph_tools import *
from datasets.utils.extract_ordered_games import *
from tst.tst_weight_conversion.old_newman import *

def iterate_equation_newman_weighted_verbose(player_idx, pi_values, games_with_players):
    a = b = 1.0 / (pi_values[player_idx] + 1.0)
    print(f"Initial a, b: {a}, {b}")

    for K, position, game, weight in games_with_players:
        print(f"Game: {game}, Position: {position}, Weight: {weight}")

        score_sums = [pi_values[p] for p in game]
        cumulative_sum = [0] * (K + 1)

        # Calculate cumulative sums
        for j in range(1, K + 1):
            cumulative_sum[j] = cumulative_sum[j - 1] + score_sums[j - 1]
        print(f"Cumulative sum: {cumulative_sum}")

        if position < K - 1:
            tmp1 = cumulative_sum[K] - cumulative_sum[position + 1]
            tmp2 = tmp1 + score_sums[position]
            if tmp2 != 0:
                a += weight * (tmp1 / tmp2)

        for v in range(position):
            tmp = cumulative_sum[K] - cumulative_sum[v]
            if tmp != 0:
                b += weight * (1.0 / tmp)
                
    print(f"Updated a: {a}")
    print(f"Updated b: {b}")

    result = a / b
    print(f"Result a / b: {result}")
    return result

def iterate_equation_newman_verbose(s, scores, bond_matrix):
    a = b = 1.0 / (scores[s] + 1.0)
    print(f"Initial a, b: {a}, {b}")

    if s in bond_matrix:
        for K in bond_matrix[s]:
            for r in bond_matrix[s][K]:
                if r < K - 1:
                    for t in range(0, len(bond_matrix[s][K][r])):
                        tmp1 = tmp2 = 0.0
                        for q in range(r, K):
                            if q > r:
                                tmp1 += scores[bond_matrix[s][K][r][t][q]]
                            tmp2 += scores[bond_matrix[s][K][r][t][q]]
                        if tmp2 != 0:
                            a += tmp1 / tmp2

                for t in range(0, len(bond_matrix[s][K][r])):
                  for v in range(0, r):
                      tmp = 0.0
                      for q in range(v, K):
                          tmp += scores[bond_matrix[s][K][r][t][q]]
                      b += 1.0 / tmp
                        
    print(f"Updated a: {a}")
    print(f"Updated b: {b}")

    result = a / b
    print(f"Result a / b: {result}")
    return result


def synch_solve_equations_old_verbose(bond_matrix, max_iter, pi_values, method, sens=1e-6):

    x, y, z = [], [], []
    scores = {}

    for n in pi_values:
        # scores[n] = float(np.exp(logistic.rvs(size=1)[0]))
        scores[n] = 1.0
   
    normalize_scores_old(scores)
    
    list_of_nodes = list(scores.keys())
    
    
    err = 1.0
    rms = N = 0.0
    for n in scores:
        if n != 'f_p':
            N += 1.0
            rms += (scores[n]-pi_values[n])*(scores[n]-pi_values[n])
    rms = np.sqrt(rms/N)

    x.append(0)
    y.append(rms)
    z.append(err)
    
    iteration = 0
    total_ratings = []
    while iteration < max_iter and err > sens:
        
        err = 0.0
        tmp_scores = {}


        for s in scores:
            tmp_scores[s] = method(s, scores, bond_matrix)

            
                            
        normalize_scores_old(tmp_scores)
        total_ratings.append(sum(tmp_scores.values()))

        for s in tmp_scores:
            if abs(tmp_scores[s]-scores[s]) > err:
                err = abs(tmp_scores[s]-scores[s])
            scores[s] = tmp_scores[s]
                
        iteration += 1
        
        rms = N = 0.0
        for n in scores:
            N += 1.0
            rms += (scores[n]-pi_values[n])*(scores[n]-pi_values[n])
        rms = np.sqrt(rms/N)

        x.append(iteration)
        y.append(rms)
        z.append(err)
   
 
    return scores, iteration, total_ratings

def synch_solve_equations_verbose(bond_matrix, max_iter, pi_values, method, sens=1e-6):
    players = np.array(list(pi_values.keys()))
    scores = np.ones(len(pi_values))
    normalize_scores_numpy(scores)


   
    err = 1.0
    iteration = 0
    total_ratings = []
    while iteration < max_iter and err > sens:
        err = 0
        # tmp_scores = scores.copy()
        tmp_scores = np.ones(len(pi_values))

        for s in range(len(scores)):
            if s in bond_matrix:
                games_with_player = bond_matrix[s]
                tmp_scores[s] = method(s, scores, games_with_player)
        
        normalize_scores_numpy(tmp_scores)
        total_ratings.append(np.sum(tmp_scores))

        err = np.max(np.abs(tmp_scores - scores))
        scores = tmp_scores.copy()


        iteration += 1


    final_scores = {players[i]: scores[i] for i in range(len(players))}
    return final_scores, iteration, total_ratings


def profile_test_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()
    return profiler

def print_profile_stats(profiler, sort_by='cumtime', top_n=10):
    stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
    stats.print_stats(top_n)



# ====================== begin unit tests ====================== 
def test_normalization():
    pi_values = {0: 1.6051959620602516, 1: 3.92363705696059, 2: 1.2523994298630128, 3: 0.4153304511746671, 4: 0.5847687524788774, 5: 0.9958778165275288, 6: 0.9958778165275288, 7: 1.5052678839616134, 8: 0.9054263070600583, 9: 0.3861742392385659}
    assert normalize_scores_numpy(np.array(list(pi_values.values()))) == normalize_scores_old(pi_values)

def test_iteration():
    data, pi_values = generate_model_instance(10, 100, 2, 2)
    player_idx = 0

    weighted_test = convert_games_to_dict(data)
    weighted_bin = binarize_data_weighted(weighted_test)
    weighted_bond_matrix = create_hypergraph_from_data_weight(weighted_test)
    games_with_player = weighted_bond_matrix[player_idx]

    bin = binarize_data_old(data)
    bond = create_hypergraph_from_data_old(data)

    result_weighted = iterate_equation_newman_weighted(player_idx, pi_values, games_with_player)
    result_standard = iterate_equation_newman_old(player_idx, pi_values, bond)

    assert np.isclose(result_standard,result_weighted, atol=1e-10)

def test_small_case():
    player_idx = 0
    pi_values = [1.22222222299999, 2.777777777, 3.111111, 4.222222]  
    test_games = [(0,1), (2,0), (0,2), (0,1), (3,0)]

    weighted_test = convert_games_to_dict(test_games)
    weighted_bin = binarize_data_weighted(weighted_test)
    weighted_bond_matrix = create_hypergraph_from_data_weight(weighted_bin)
    games_with_player = weighted_bond_matrix[player_idx]

    bin = binarize_data_old(test_games)
    bond = create_hypergraph_from_data_old(bin)

    scores = [1.22222222299999, 2.777777777, 3.111111, 4.222222] 
  
    
    print("Testing iterate_equation_newman_weighted")
    result_weighted = iterate_equation_newman_weighted_verbose(player_idx, pi_values, games_with_player)
    print(f"Result (Weighted): {result_weighted}")
    
    print("\nTesting iterate_equation_newman")
    result_standard = iterate_equation_newman_verbose(player_idx, scores, bond)
    print(f"Result (Standard): {result_standard}")

    assert result_standard == result_weighted

def test_synch_solve():

    data, pi_values = generate_model_instance(10, 10, 2, 2)
    
    weighted_data = convert_games_to_dict(data)
    weighted_bin = binarize_data_weighted(weighted_data)
    weighted_bond_matrix = create_hypergraph_from_data_weight(weighted_bin)

    bin = binarize_data_old(data)
    bond = create_hypergraph_from_data_old(bin)

    weighted_newman, weighted_iter, tot1 = synch_solve_equations_verbose(weighted_bond_matrix, 1000, pi_values, iterate_equation_newman_weighted)
    norm_newman, iter, tot2 = synch_solve_equations_old_verbose(bond, 1000, pi_values, iterate_equation_newman_old)

    assert weighted_iter == iter, f"Iteration mismatch: weighted_iter={weighted_iter}, iter={iter}"
        
    if weighted_iter != iter:
        for rep in range(len(tot1)):
            print(f"REP: {rep}, WEIGHTED: {tot1[rep]}, NORMAL: {tot2[rep]}, DIFF: {tot1[rep] - tot2[rep]}")

    assert len(weighted_newman) == len(norm_newman), f"Length mismatch: weighted_newman={len(weighted_newman)}, norm_newman={len(norm_newman)}"
    assert all(np.isclose(weighted_newman[player], norm_newman[player], atol=1e-10) for player in norm_newman), "Values mismatch between weighted_newman and norm_newman"

def test_small_batch():
    data, pi_values = generate_model_instance(20, 20, 2, 2)

    weighted_data = convert_games_to_dict(data)
    
    new_newman = compute_predicted_ratings_BT(weighted_data, pi_values)
    old_newman = compute_predicted_ratings_BT_old(data, pi_values)
  
    
    assert all(isinstance(x, float) for x in new_newman.values())
                          
def test_weighted_newman():
    data, pi_values = generate_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_newman = compute_predicted_ratings_BT(weighted_data, pi_values)
    old_newman = compute_predicted_ratings_BT_old(data, pi_values)
  
    
    assert len(old_newman) == len(new_newman)
    assert all(np.isclose(old_newman[player], new_newman[player], atol=1e-10) for player in new_newman), print(old_newman, new_newman)

def test_bin():
    data, pi_values = generate_model_instance(10, 10, 2, 2)

    weighted_data = convert_games_to_dict(data)
    
    new_newman = compute_predicted_ratings_BT(weighted_data, pi_values)
    old_newman = compute_predicted_ratings_BT_old(data, pi_values)

    for player in pi_values:
        print(new_newman[player], old_newman[player])

    assert len(old_newman) == len(new_newman)
    assert all(np.isclose(old_newman[player], new_newman[player], atol=1e-10) for player in new_newman)

def test_binarized_leadership(): 
    data, pi_values = generate_leadership_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_newman = compute_predicted_ratings_BT_leadership(weighted_data, pi_values)
    old_newman = compute_predicted_ratings_BT_leadership_old(data, pi_values)
  

    assert len(old_newman) == len(new_newman)
    assert all(np.isclose(old_newman[player], new_newman[player], atol=1e-10) for player in new_newman)

def test_weighted_ho():
    data, pi_values = generate_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_newman = compute_predicted_ratings_HO_BT(weighted_data, pi_values)
    old_newman = compute_predicted_ratings_HO_BT_old(data, pi_values)
  
    
    assert len(old_newman) == len(new_newman)
    assert all(np.isclose(old_newman[player], new_newman[player], atol=1e-10) for player in new_newman)



def test_weighted_hol():
    data, pi_values = generate_leadership_model_instance(100, 500, 4, 4)

    weighted_data = convert_games_to_dict(data)
    
    new_newman = compute_predicted_ratings_HOL_BT(weighted_data, pi_values)
    old_newman = compute_predicted_ratings_HOL_BT_old(data, pi_values)
  
   
    assert len(old_newman) == len(new_newman)
    assert all(np.isclose(old_newman[player], new_newman[player], atol=1e-10) for player in new_newman)




def main():
    data, pi_values = generate_model_instance(3000, 3000, 4, 4)
    weighted_data = convert_games_to_dict(data)
    
    profiler = profile_test_function(compute_predicted_ratings_BT_leadership, weighted_data, pi_values)
    print_profile_stats(profiler)

    profiler = profile_test_function(compute_predicted_ratings_BT_leadership_old, data, pi_values)
    print_profile_stats(profiler)

if __name__ == '__main__':
    main()