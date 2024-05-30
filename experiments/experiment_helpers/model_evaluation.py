
import sys
import os
import numpy 
import random
import matplotlib.pyplot as plt 

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.solvers import *
from experiments.experiment_helpers.metrics import * 
from src.syntetic import *
from sklearn.model_selection import train_test_split


def evaluate_model_likelihood(N,M,K1,K2):

    all_results = []

    for _ in range(10):

        pi_values, data = generate_model_instance(N, M, K1, K2)

        random.shuffle(data)
        
        for train_size in np.logspace(-2, 0, endpoint=False, num=25):

            training_set, testing_set = split_games(data,train_size)
            
            hyper_graph_pred, graph_pred = compute_predicted_rankings(training_set=training_set, ground_truth_ratings=pi_values)

            hyper_graph_likelihood = compute_likelihood(hyper_graph_pred, pi_values, testing_set)
            graph_likelihood = compute_likelihood(graph_pred, pi_values, testing_set)

            all_results.append({
                'loss_measurement': 'Likelihood',
                'train_size': train_size,
                'N': N,
                'M': M,
                'K1': K1,
                'K2': K2,
                'std_bt_likelihood': graph_likelihood,
               
                'ho_bt_likelihood': hyper_graph_likelihood,
               
            })


    return all_results

def split_games(games, train_size):
    
    sampled_games = int(train_size * len(games))

    training_set = games[:sampled_games]
    testing_set = games[sampled_games:]

    return training_set, testing_set

def compute_predicted_rankings(training_set, ground_truth_ratings):

    # Create hypergraph
    bond_matrix = create_hypergraph_from_data(training_set)

    # Standard graph
    bin_data = binarize_data(training_set)
    bin_bond_matrix = create_hypergraph_from_data(bin_data)
    
    # predict ranks based on subset of games
    predicted_hyper_graph_scores, _ = synch_solve_equations(bond_matrix, 500, ground_truth_ratings, 'newman', sens=1e-6)
    predicted_graph_scores, _ = synch_solve_equations(bin_bond_matrix, 500, ground_truth_ratings, 'newman', sens=1e-6)

    return predicted_hyper_graph_scores, predicted_graph_scores


def compute_likelihood(pred_ranking, true_ranking, testing_set):

    game_likelihood = []
    for game in testing_set:
        likelihood = recursive_probability_estimation(pred_ranking, game)
        null_likelihood = recursive_probability_estimation(true_ranking, game)

        game_likelihood.append(likelihood / null_likelihood)


    return np.mean(game_likelihood)

def compute_likelihood_shuffled(pred_ranking, true_ranking, testing_set):

    game_likelihood = []
    for game in testing_set:
        shuffled_game = game.copy()
        random.shuffle(shuffled_game)
        
        likelihood = recursive_probability_estimation(pred_ranking, game)
        null_likelihood = recursive_probability_estimation(pred_ranking, shuffled_game)

        game_likelihood.append(likelihood / null_likelihood)


    return np.mean(game_likelihood)


def recursive_probability_estimation(pred_rating, game, total_prob_estimation=1):

    player_rankings = [pred_rating[player] for player in game]
    highest_rank = player_rankings[0]
    total_ratings = sum(player_rankings)

    total_prob_estimation *= highest_rank/total_ratings

    if len(player_rankings) > 2: 
        return recursive_probability_estimation(pred_rating, game[1:], total_prob_estimation)
    else:
        return total_prob_estimation



if __name__ == '__main__': 


    N = 500 
    M = 1000
    K1 = 5 
    K2 = 10

 

    true_distribution, ho_distribution, std_distribution = evaulate_model_likelihood(N, M, K1, K2)
    print(np.mean(true_distribution), np.std(true_distribution))
    print(np.mean(ho_distribution), np.std(ho_distribution))
    print(np.mean(std_distribution), np.std(std_distribution))