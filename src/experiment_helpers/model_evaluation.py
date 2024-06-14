
import sys
import os
import csv
import random


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.models.ho_hol_newman import *
from exp.experiment_helpers.metrics import * 
from src.synthetic import *


def split_games(games, train_size):
    
    sampled_games = int(train_size * len(games))

    training_set = games[:sampled_games]
    testing_set = games[sampled_games:]

    return training_set, testing_set

def compute_predicted_rankings_ho(training_set, ground_truth_ratings): 

    bond_matrix = create_hypergraph_from_data (training_set)
    bin_data = binarize_data (training_set)
    bin_bond_matrix = create_hypergraph_from_data (bin_data)

    # predict ranks based on subset of games
    predicted_ho_scores, _ = synch_solve_equations(bond_matrix, 1000, ground_truth_ratings, 'newman', sens=1e-6)
    predicted_hol_scores, _ = synch_solve_equations (bond_matrix, 1000, ground_truth_ratings, 'newman_leadership', sens=1e-6)
    predicted_std_scores, _ = synch_solve_equations(bin_bond_matrix, 1000, ground_truth_ratings, 'newman', sens=1e-6)

    return predicted_ho_scores, predicted_hol_scores, predicted_std_scores

def compute_predicted_rankings_hol(training_set, ground_truth_ratings):
    bond_matrix = create_hypergraph_from_data (training_set)
    bin_data = binarize_data_leadership (training_set)
    bin_bond_matrix = create_hypergraph_from_data (bin_data)

    predicted_ho_scores, _ = synch_solve_equations(bond_matrix, 1000, ground_truth_ratings, 'newman', sens=1e-6)
    predicted_hol_scores, _ = synch_solve_equations (bond_matrix, 1000, ground_truth_ratings, 'newman_leadership', sens=1e-6)
    predicted_std_scores, _ = synch_solve_equations(bin_bond_matrix, 1000, ground_truth_ratings, 'newman', sens=1e-6)

    return predicted_ho_scores, predicted_hol_scores, predicted_std_scores

def benchmark_ho(training_set, testing_set, pi_values):
    predicted_ho_scores, predicted_hol_scores, predicted_std_scores = compute_predicted_rankings_ho(training_set=training_set, ground_truth_ratings=pi_values)

    ho_likelihood = compute_likelihood(predicted_ho_scores, testing_set)
    hol_likelihood = compute_likelihood(predicted_hol_scores, testing_set)
    std_likelihood = compute_likelihood(predicted_std_scores, testing_set)

    return ho_likelihood, hol_likelihood, std_likelihood

def benchmark_hol(training_set, testing_set, pi_values):
    predicted_ho_scores, predicted_hol_scores, predicted_std_scores = compute_predicted_rankings_hol(training_set=training_set, ground_truth_ratings=pi_values)

    ho_likelihood = compute_leadership_likelihood(predicted_ho_scores, testing_set)
    hol_likelihood = compute_leadership_likelihood(predicted_hol_scores, testing_set)
    std_likelihood = compute_leadership_likelihood(predicted_std_scores, testing_set)

    return ho_likelihood, hol_likelihood, std_likelihood


def generate_and_benchmark_ho_model(N, M, K1, K2, train_size=.8):
    pi_values, data = generate_model_instance(N, M, K1, K2)

    random.shuffle(data)

    training_set, testing_set = split_games(data, train_size)
    return benchmark_ho(training_set, testing_set, pi_values)


def generate_and_benchmark_hol_model(N, M, K1, K2, train_size=.8): 
    pi_values, data = generate_leadership_model_instance(N, M, K1, K2)

    random.shuffle(data)

    training_set, testing_set = split_games(data, train_size)
    return benchmark_hol(training_set, testing_set, pi_values)
