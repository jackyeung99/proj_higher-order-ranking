import os 
import sys
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models import *

MODEL_FUNCTIONS = {
    'newman': compute_predicted_ratings_std,
    'newman_leadership': compute_predicted_ratings_std_leadership,
    'higher_order_newman': compute_predicted_ratings_ho,
    'higher_order_leadership': compute_predicted_ratings_hol,
    'spring_rank': compute_predicted_ratings_spring_rank,
    'spring_rank_leadership': compute_predicted_ratings_spring_rank_leadership,
    'page_rank': compute_predicted_ratings_page_rank,
    'page_rank_leadership': compute_predicted_ratings_page_rank_leadership,
    'point_wise': compute_point_wise_ratings
}

def get_predictions(model, training_set, pi_values):
    if model in MODEL_FUNCTIONS:
        return MODEL_FUNCTIONS[model](training_set, pi_values)
    elif model == 'tensor_flow':
        # return compute_predicted_rankings_tensor_flow(training_set, pi_values)
        pass

def run_models(train_data, test_data, pi_values, leadership=False):
    likelihoods_dict = {'Game': list(range(len(test_data)))}
    for model in MODEL_FUNCTIONS:
        predicted_rankings = get_predictions(model, train_data, pi_values)
        
        if leadership:
            game_likelihoods = compute_leadership_likelihood(predicted_rankings, test_data)
        else: 
            game_likelihoods = compute_likelihood(predicted_rankings, test_data)

        likelihoods_dict[model] = game_likelihoods

    df = pd.DataFrame(likelihoods_dict)
    return df

def split_games(games, train_size):
    
    sampled_games = int(train_size * len(games))

    training_set = games[:sampled_games]
    testing_set = games[sampled_games:]

    return training_set, testing_set



def calculate_percentages(df, compared_axis):
    total_rows = len(df)
    if total_rows == 0:
        return {}

    # Ensure the compared_axis is valid
    if compared_axis < 0 or compared_axis >= df.shape[1]:
        raise ValueError("Invalid compared_axis index")

    comparisons = {
        df.columns[col]: (df.iloc[:, col] > df.iloc[:, compared_axis]).sum() / total_rows 
        for col in range(df.shape[1]) if col != compared_axis
    }

    return comparisons

def calculate_column_means(df, compared_axis):
    if df.empty:
        return {}

    # Ensure the compared_axis is valid
    if compared_axis < 0 or compared_axis >= df.shape[1]:
        raise ValueError("Invalid compared_axis index")

    means = {
        df.columns[col]: (df.iloc[:, col] - df.iloc[:, compared_axis]).mean() 
        for col in range(df.shape[1]) if col != compared_axis
    }

    return means
