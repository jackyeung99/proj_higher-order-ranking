import os 
import sys
import pandas as pd
import random
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models import *
from src.utils.metrics import measure_leadership_likelihood, measure_likelihood, measure_rho, measure_rms, measure_tau

BASE_FUNCTIONS = {
    'HO_BT': compute_predicted_ratings_HO_BT,
    'HOL_BT': compute_predicted_ratings_HOL_BT,
    'BIN': compute_predicted_ratings_BIN,
    'BINL': compute_predicted_ratings_BINL,
}

COMPARISON_MODELS = {
    'Spring_Rank': compute_predicted_ratings_spring_rank,
    'Page_Rank': compute_predicted_ratings_page_rank,
    'Point_Wise': compute_point_wise_ratings
}

MODEL_FUNCTIONS = {**BASE_FUNCTIONS, **COMPARISON_MODELS}

def get_predictions(model, training_set, pi_values, verbose=False):
    if model in MODEL_FUNCTIONS:
        if verbose:
            return MODEL_FUNCTIONS[model](training_set, pi_values, verbose=True)
        else:
            return MODEL_FUNCTIONS[model](training_set, pi_values)

    elif model == 'tensor_flow':
        # return compute_predicted_rankings_tensor_flow(training_set, pi_values)
        pass

def run_models_synthetic(train_data, test_data, pi_values):
    model_performance = []
    for model in BASE_FUNCTIONS:
        predicted_ratings, iter = get_predictions(model, train_data, pi_values,verbose=True)
       
        
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
            'tau': tau,
            'iteration': len(iter)
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

    # training_set = games[:sampled_games]
    # testing_set = games[sampled_games:]

    
    test = len(games) - sampled_games
    testing_set = games[:test]
    training_set = games[test:]

    return training_set, testing_set


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

