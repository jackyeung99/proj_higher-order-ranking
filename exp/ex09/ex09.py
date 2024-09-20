import os 
import sys

import pandas as pd 

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *
from src.models.BradleyTerry import *
from tst.tst_weight_conversion.old_newman import * 

def compute_predicted_ratings_HO_BT_info(training_set, pi_values): 
    bond_matrix = create_hypergraph_from_data_weight(training_set)
    predicted_ho_scores, iter = synch_solve_equations(bond_matrix, 10000, pi_values, iterate_equation_newman_weighted, sens=1e-6)
    return predicted_ho_scores, iter



def iteration_at_convergence(K):
    data, pi_values = generate_weighted_model_instance(1000, 1000, K , K)
    predicted_scores, iteration = compute_predicted_ratings_HO_BT_info(data, pi_values)
    return iteration


def hyper_edge_iteration(repetitions, out_file_dir, K_values):
    os.makedirs(out_file_dir, exist_ok=True)

    values = []
    for K in K_values:
        for rep in range(repetitions):
            iteration = iteration_at_convergence(K)
            values.append({"K": K, "Rep": rep, "Iterations_to_converge": iteration}) 

    df = pd.DataFrame(values)
    path = os.path.join(out_file_dir, 'Hyperedge_iteration_count.csv')
    df.to_csv(path)


if __name__ == '__main__':
    K_values = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    reps = 1000
    hyper_edge_iteration(reps, out_dir, K_values)







