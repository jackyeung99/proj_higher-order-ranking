import os
import sys
import csv


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.solvers import *
from experiments.experiment_helpers.metrics import * 
from experiments.experiment_helpers.model_evaluation import * 
from src.syntetic import *


def evaluate_model_likelihood(tested_param, N, M, K1, K2):

    all_results = []

    for n in range(50):

        pi_values, data = generate_model_instance(N, M, K1, K2)

        random.shuffle(data)
        
        for train_size in np.logspace(-2, 0, endpoint=False, num=25):

            training_set, testing_set = split_games(data,train_size)
            
            hyper_graph_pred, graph_pred = compute_predicted_rankings(training_set=training_set, ground_truth_ratings=pi_values)

            hyper_graph_likelihood = compute_likelihood(hyper_graph_pred, testing_set)
            graph_likelihood = compute_likelihood(graph_pred, testing_set)

            all_results.append({
                'tested_parameter': tested_param,
                'M': M,
                'N': N,
                'K1': K1,
                'K2' : K2,
                'train_size': train_size,
                'std_bt_likelihood': graph_likelihood,
                'ho_bt_likelihood': hyper_graph_likelihood,
                'repetition': n 
                })
            
    return all_results
        
def run_experiments(M_values, K1_values, K2_values, headers, out_file):

    N = 1000
    M = 1500
    K1 = 2
    K2 = 6 

    # for m in M_values:
    #     results = evaluate_model_likelihood('games', N, m, K1, K2)
    #     save_results_to_csv(out_file, headers, results)

    # for K1 in K1_values:
    #     K2 = K1 + 1
    #     results = evaluate_model_likelihood('hyperedge_size', N, M, K1, K2) 
    #     save_results_to_csv(out_file, headers, results)

    for K2 in K2_values:
        results = evaluate_model_likelihood('K_gap', N, M, K1, K2)
        save_results_to_csv(out_file, headers, results)

if __name__ == '__main__':
    
    headers = ['tested_parameter', 'M', 'N', 'K1','K2', 'train_size', 'std_bt_likelihood', 'ho_bt_likelihood', 'repetition']

    M_values = [500, 1000, 2000] 
    K1_values = [2, 4, 8, 16]
    K2_values = [2, 4, 8, 16] 

    out_file = 'ex01_results.csv'

    run_experiments(M_values, K1_values, K2_values, headers, out_file)