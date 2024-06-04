import os
import sys
import csv

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.solvers import *
from experiments.experiment_helpers.metrics import * 
from experiments.experiment_helpers.model_evaluation import * 
from src.syntetic import *
from src.file_readers import * 


def evaluate_model_prediction(dataset, pi_values, data):

    all_results = []
    for n in range(50):

        random.shuffle(data)
        training_set, testing_set = split_games(data, .8)
        
        bond_matrix = create_hypergraph_from_data(training_set)

        # Standard graph
        bin_data = binarize_data(training_set)
        bin_bond_matrix = create_hypergraph_from_data(bin_data)
        
        # predict ranks based on subset of games
        scores_ho, _ = synch_solve_equations (bond_matrix, 1000, pi_values, 'newman', sens=1e-6)
        # scores_hol, _ = synch_solve_equations (bond_matrix, 1000, pi_values, 'newman_leadership', sens=1e-6)
        scores_std, _ = synch_solve_equations (bin_bond_matrix, 1000, pi_values, 'newman', sens=1e-6)

        hyper_graph_likelihood = compute_likelihood(scores_ho, pi_values, testing_set)
        # hol_likelihood = compute_likelihood(scores_hol, pi_values, testing_set)
        graph_likelihood = compute_likelihood(scores_std, pi_values, testing_set)

        
        all_results.append({
            'dataset': dataset, 
            'ho_bt_likelihood': hyper_graph_likelihood, 
            # 'hol_likelihood': hol_likelihood,
            'std_bt_likelihood': graph_likelihood,
            'repetition': n
            })
            
    return all_results


def run_experiments(out_file, headers):
    def read_and_evaluate(file_path, read_data_function, data_label):
        data, pi_values = read_data_function(file_path)
        results = evaluate_model_prediction(data_label, pi_values, data)
        save_results_to_csv(out_file, headers, results)
    
    read_and_evaluate('data/fifa_wc.txt', read_data_fifa, 'fifa')
    read_and_evaluate('data/authorships.txt', read_data_authors, 'authors')
    read_and_evaluate('data/cl_data.txt', read_data_ucl, 'ucl')


if __name__ == '__main__':

    out_file = os.path.join(os.path.dirname(__file__), 'ex03_results.csv')
    headers = ['dataset', 'ho_bt_likelihood', 'hol_likelihood', 'std_bt_likelihood', 'repetition']
    run_experiments(out_file, headers) 