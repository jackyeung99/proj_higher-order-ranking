import numpy 
import random
from src.solvers import *


def evaulate_training_sizes(N,M,K1,K2, loss_measurement ='rms'):

    all_results = []

    pi_values, data = generate_model_instance(N, M, K1, K2)

    for train_size in np.arange(0.1, 1.1, 0.1):
        
        training_set, testing_set = split_games(data, train_size)

        hyper_graph_pred, graph_pred = compute_rankings(training_set=training_set, ground_truth_ratings=pi_values)

        hyper_graph_likelihood = compute_likelihood(hyper_graph_pred, testing_set)
        graph_likelihood = compute_likelihood(graph_pred, testing_set)


        all_results.append({
            'loss_measurement': loss_measurement,
            'train_size': train_size,
            'N': N,
            'M': M,
            'K1': K1,
            'K2': K2,
            'graph_accuracy': graph_likelihood,
            'hyper_graph_accuracy': hyper_graph_likelihood
        })

    return all_results

def split_games(games, train_size):
    
    sampled_games = train_size * len(train_size)

    training_set = random.sample(games, sampled_games)
    testing_set = [game for game in games if game not in training_set]

    return training_set, testing_set

def compute_rankings(training_set, ground_truth_ratings):

    # Create hypergraph
    bond_matrix = create_hypergraph_from_data(training_set)

    # Standard graph
    bin_data = binarize_data(training_set)
    bin_bond_matrix = create_hypergraph_from_data(bin_data)
    
    # predict ranks based on subset of games
    predicted_hyper_graph_scores , _ = synch_solve_equations(bond_matrix, 500, ground_truth_ratings, 'newman', sens=1e-6)
    predicted_graph_scores, _ = synch_solve_equations(bin_bond_matrix, 500, ground_truth_ratings, 'newman', sens=1e-6)


    return predicted_hyper_graph_scores, predicted_graph_scores



def compute_likelihood(pred_ranking, true_ranking, test_set):
    
    game_likelihood = []
    for game in test_set:
        likelihood = recursive_mle_estimation(pred_ranking, game)
        expected_likelihood = recursive_mle_estimation(true_ranking, game)
        game_likelihood.append(likelihood / expected_likelihood)

    return np.mean(game_likelihood), np.std(game_likelihood)
        
def recursive_mle_estimation(pred_ranking, game, total_mle_estimation=0):
    
    player_rankings = [pred_ranking[player] for player in game]
    highest_rank = player_rankings[0]
    total_ratings = sum(player_rankings)
    total_mle_estimation += highest_rank/total_ratings

    if len(player_rankings) > 2: 
        recursive_mle_estimation(pred_ranking, game[1:], total_mle_estimation)
    else:
        return total_mle_estimation

