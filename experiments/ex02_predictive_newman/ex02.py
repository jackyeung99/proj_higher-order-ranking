import os
import sys
import csv

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.solvers import *
from experiments.experiment_helpers.metrics import * 
from experiments.experiment_helpers.model_evaluation import * 
from src.syntetic import *


def evaluate_model_prediction(N,M,K1,K2):

    all_results = []

    for n in range(50):

        pi_values, data = generate_model_instance(N, M, K1, K2)

        random.shuffle(data)
        
        training_set, testing_set = split_games(data, .8)
        
        hyper_graph_pred, graph_pred = compute_predicted_rankings(training_set=training_set, ground_truth_ratings=pi_values)

        hyper_graph_likelihood = compute_likelihood(hyper_graph_pred, pi_values, testing_set)
        graph_likelihood = compute_likelihood(graph_pred, pi_values, testing_set)

        
        all_results.append({
            'M': M,
            'N': N,
            'K1': K1,
            'K2' : K2,
            'std_bt_likelihood': graph_likelihood,
            'ho_bt_likelihood': hyper_graph_likelihood, 
            'repetition': n,
            })
            
    return all_results


def run_experiments(out_file, headers, N, M_values, K1_values): 
    for m in M_values:
        M = int(m)
        for K1 in K1_values:
            K2 = K1 + 1
            results = evaluate_model_prediction(N, M, K1, K2)
            save_results_to_csv(out_file, headers, results)


if __name__ == '__main__':

    N = 1000
    K1_values = [2, 6, 10, 14, 18]
    M_values = np.logspace(6, 14, num=8, endpoint=True, base=2)
    
    headers = ['M', 'N', 'K1', 'K2', 'std_bt_likelihood', 'ho_bt_likelihood', 'repetition']
    out_file = 'ex02_results.csv'
    run_experiments(out_file, headers, N, M_values, K1_values)