
import sys
import os
import csv
import random


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.solvers import *
from experiments.experiment_helpers.metrics import * 
from src.syntetic import *


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
    
def save_results_to_csv(filename, headers, results):
    file_path = os.path.join(os.getcwd(), filename)
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for result in results:
            writer.writerow(result)



