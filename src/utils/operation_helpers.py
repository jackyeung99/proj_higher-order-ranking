import os 
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models import *


def split_games(games, train_size):
    
    sampled_games = int(train_size * len(games))

    training_set = games[:sampled_games]
    testing_set = games[sampled_games:]

    return training_set, testing_set


def get_predictions(model, training_set, pi_values):

    if model == 'newman':
        return compute_predicted_ratings_std(training_set, pi_values)
    elif model == 'newman_leadership':
        return compute_predicted_ratings_std_leadership(training_set, pi_values)
    elif model == 'higher_order_newman':
        return compute_predicted_ratings_ho(training_set, pi_values)
    elif model == 'higher_order_leadership':
        return compute_predicted_ratings_hol(training_set, pi_values)
    elif model == 'spring_rank':
        return compute_predicted_ratings_spring_rank(training_set, pi_values)
    elif model == 'spring_rank_leadership':
        return compute_predicted_ratings_spring_rank_leadership(training_set, pi_values)
    elif model == 'page_rank':
        return compute_predicted_ratings_page_rank(training_set, pi_values)
    elif model == 'page_rank_leadership':
        return compute_predicted_ratings_page_rank_leadership(training_set, pi_values)
    elif model == 'tensor_flow':
        # return compute_predicited_rankings_tensor_flow(training_set, pi_values)
        pass
    elif model == 'point_wise':
        return compute_point_wise_ratings(training_set, pi_values)


def run_models(train_data, test_data, pi_values):

    models = ['newman', 'newman_leadership', 'higher_order_newman', 'higher_order_leadership', 'spring_rank', 'spring_rank_leadership', 'page_rank', 'page_rank_leadership', 'point_wise']
   
    likelihoods_dict = {'Game': list(range(len(test_data)))}
    for model in models:
        predicted_rankings = get_predictions(model, train_data, pi_values)
        game_likelihoods = compute_likelihood(predicted_rankings, test_data)
        likelihoods_dict[model] = game_likelihoods

    df = pd.DataFrame(likelihoods_dict)

    return df
    # df.to_csv(file_name, index=False)