import sys
import os 

import random 
import pandas as pd
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *
from tst.tst_weight_conversion.old_newman import iterate_equation_newman_old
from src.models.zermello import *




# Modified Solver function  to Keep Track of iterations for Genralized Newman 
def synch_solve_equations_info(bond_matrix, max_iter, pi_values, method, sens=1e-6):
    scores = {}
    logistic_distribution = np.sqrt(np.exp(logistic.rvs(size=len(pi_values))))
    scores = {n: logistic_distribution[n-1] for n in pi_values}
    normalize_scores_old(scores)

    info = {}
    err = 1.0
    iteration = 0
    while iteration < max_iter and err > sens:
        
        err = 0.0
        tmp_scores = {}


        for s in scores:
            tmp_scores[s] = method(s, scores, bond_matrix)

                 
        normalize_scores_old(tmp_scores)

        for s in tmp_scores:
            cur_err = abs(np.log(tmp_scores[s])-np.log(scores[s]))
            if cur_err > err:
                err = cur_err
            scores[s] = tmp_scores[s]


        iteration += 1
        info[iteration] = err

   
    return scores, info


def compute_predicted_ratings_HO_BT_info_random(training_set, pi_values, max_iter=10000): 
    # Using un-weighted newman iterative schema such that a random shuffle of the data can occur 
    bond_matrix = create_hypergraph_from_data_old(training_set)
    predicted_ho_scores, info  = synch_solve_equations_info(bond_matrix, max_iter, pi_values, iterate_equation_newman_old, sens=1e-6)
 
    return predicted_ho_scores, info


def compute_predicted_ratings_plackett_luce_random(training_set, pi_values, max_iter=10000): 
    bond_matrix = create_hypergraph_from_data_old(training_set)
    predicted_ho_scores, info = synch_solve_equations_info(bond_matrix, max_iter, pi_values, iterate_equation_zermelo_new, sens=1e-6)
 
    return predicted_ho_scores, info


# Run both models keeping track of error for iterations
def test_convergence(un_weighted_data, pi_values, max_iter):
    _, HO_info = compute_predicted_ratings_HO_BT_info_random(un_weighted_data, pi_values, max_iter)
    _, PL_info = compute_predicted_ratings_plackett_luce_random(un_weighted_data, pi_values, max_iter)

    ho_errors = np.zeros(max_iter)
    pl_errors = np.zeros(max_iter)

    for i in range(max_iter):
        if i in HO_info:
            ho_errors[i] = HO_info[i]
        if i in PL_info:
            pl_errors[i] = PL_info[i]

    return ho_errors, pl_errors

# for each repetition obtain error for each iteration
def save_convergence_data(file_name, data, pi_values, max_iter = 10000):
    ho_errors, pl_errors = test_convergence(data, pi_values, max_iter)
    
    data = {
        'Iteration': np.arange(max_iter),
        'Avg_HO_Error': ho_errors,
        'Avg_PL_Error': pl_errors,
    }

    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)


