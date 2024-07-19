import os 
import sys
import pandas as pd
import random
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models import *
from src.utils.metrics import *

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

def run_models_synthetic(train_data, test_data, pi_values):
    model_performance = []
    for model in MODEL_FUNCTIONS:
        predicted_ratings = get_predictions(model, train_data, pi_values)
        
        log_likelihoods = measure_likelihood(predicted_ratings, test_data)
        leadership_log_likelihoods = measure_leadership_likelihood(predicted_ratings, test_data)
        rms = measure_rms(predicted_ratings, pi_values)
        rho = measure_rho(predicted_ratings, pi_values)
        tau = measure_tau(predicted_ratings, pi_values)

        model_results = {
            'model': model,
            'log-likelihood': log_likelihoods,
            'leadership-log-likelihood': leadership_log_likelihoods,
            'rms': rms,
            'rho': rho,
            'tau': tau
            }
        
        model_performance.append(model_results)

    return pd.DataFrame(model_performance)

def run_models(train_data, test_data, pi_values):
    model_performance = []
    for model in MODEL_FUNCTIONS:
        predicted_ratings = get_predictions(model, train_data, pi_values)
        
        game_likelihoods = measure_likelihood(predicted_ratings, test_data)
        leadership_likelihoods = measure_leadership_likelihood(predicted_ratings, test_data)

        model_results = { 
            'model': model,
            'log-likelihoods': game_likelihoods,
            'leadership-log-likelihood': leadership_likelihoods,
            }
        
        model_performance.append(model_results)


    return pd.DataFrame(model_performance)


def split_games(games, train_size):
    
    sampled_games = int(train_size * len(games))

    training_set = games[:sampled_games]
    testing_set = games[sampled_games:]

    return training_set, testing_set

def split_weighted_dataset(dataset, train_ratio=0.8):
    expanded_list = []
    for key, weight in dataset.items():
        expanded_list.extend([key] * weight)
    
    random.shuffle(expanded_list)
    training_set, testing_set = train_test_split(expanded_list, train_size=train_ratio, random_state=None)

    weighted_training_set = convert_games_to_dict(training_set)
    weighted_testing_set = convert_games_to_dict(testing_set)
    
    return weighted_training_set, weighted_testing_set

def calculate_percentages_against_base(df, compared_column, flipped=False):
    total_rows = len(df)
    if total_rows == 0:
        return {}

    # Ensure the compared_column exists in the DataFrame
    if compared_column not in df.columns:
        raise ValueError("Invalid compared_column name")

    if flipped:
        comparisons = {
            col: (df[col] < df[compared_column]).sum() / total_rows 
            for col in df.columns 
            if col != compared_column and col not in ['rep', 'train']
        }
    else:
        comparisons = {
            col: (df[col] > df[compared_column]).sum() / total_rows 
            for col in df.columns 
            if col != compared_column and col not in ['rep', 'train']
        }

    return comparisons

def calculate_column_means_against_base(df, compared_column):
    if df.empty:
        return {}

    # Ensure the compared_column exists in the DataFrame
    if compared_column not in df.columns:
        raise ValueError("Invalid compared_column name")

    means = {
        col: (df[col] - df[compared_column]).mean() 
        for col in df.columns 
        if col != compared_column and col not in ['rep', 'train']
    }

    return means

