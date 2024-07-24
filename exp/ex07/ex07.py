import os
import sys
import matplotlib.pyplot as plt
import pandas as pd


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *
from src.models.BradleyTerry import *
from src.models.zermello import *



def synch_solve_equations_w_iterations(bond_matrix, max_iter, pi_values, method, sens=1e-10):
    info = {}
    
    
    players = np.array(list(pi_values.keys()))
    scores = np.ones(len(pi_values))
    normalize_scores_numpy(scores)

    err = 1.0
    iteration = 0
    
    while iteration < max_iter and err > sens:
        err = 0
        tmp_scores = scores.copy()

        for s in range(len(scores)):
            if s in bond_matrix:
                games_with_player = bond_matrix[s]
                tmp_scores[s] = method(s, scores, games_with_player)
        
        normalize_scores_numpy(tmp_scores)
       
        err = np.max(np.abs(tmp_scores - scores))
        scores = tmp_scores.copy()
        
        iteration += 1

        info[iteration] = err
    
        
    final_scores = {players[i]: scores[i] for i in range(len(players))}
    return final_scores, info


def compute_predicted_ratings_HO_BT_info(training_set, pi_values): 
    bond_matrix = create_hypergraph_from_data_weight(training_set)
    predicted_ho_scores, info  = synch_solve_equations_w_iterations(bond_matrix, 1000, pi_values, iterate_equation_newman_weighted, sens=1e-10)
 
    return predicted_ho_scores, info

def synch_solve_equations_w_iterations(bond_matrix, max_iter, pi_values, method, sens=1e-10):
    info = {}
    
    
    players = np.array(list(pi_values.keys()))
    scores = np.ones(len(pi_values))
    normalize_scores_numpy(scores)

    err = 1.0
    iteration = 0
    
    while iteration < max_iter and err > sens:
        err = 0
        tmp_scores = scores.copy()

        for s in range(len(scores)):
            if s in bond_matrix:
                games_with_player = bond_matrix[s]
                tmp_scores[s] = method(s, scores, games_with_player)
        
        normalize_scores_numpy(tmp_scores)
       
        err = np.max(np.abs(tmp_scores - scores))
        scores = tmp_scores.copy()
        
        iteration += 1

        info[iteration] = err
    
        
    final_scores = {players[i]: scores[i] for i in range(len(players))}
    return final_scores, info


def compute_predicted_ratings_HO_BT_info(training_set, pi_values): 
    bond_matrix = create_hypergraph_from_data_weight(training_set)
    predicted_ho_scores, info  = synch_solve_equations_w_iterations(bond_matrix, 1000, pi_values, iterate_equation_newman_weighted, sens=1e-10)
 
    return predicted_ho_scores, info

def test_convergence(leadership, max_iter):
    if leadership:
        data, pi_values = generate_leadership_model_instance(1000, 10000, 5, 5)
    else:
        data, pi_values = generate_model_instance(1000, 10000, 5, 5)

    weighted_data = convert_games_to_dict(data)
    _, ho_info = compute_predicted_ratings_HO_BT_info(weighted_data, pi_values)
        
    _, PL_info = compute_predicted_ratings_plackett_luce(data, pi_values)

    ho_errors = np.zeros(max_iter)
    pl_errors = np.zeros(max_iter)

    for i in range(max_iter):
        if i in ho_info:
            ho_errors[i] = ho_info[i]
        if i < len(PL_info[1]):
            pl_errors[i] = PL_info[1][i]

    return ho_errors, pl_errors

def average_convergence(reps, leadership, output_file, max_iter=1000):
    os.makedirs(output_file, exist_ok=True)

    ho_errors_sum = np.zeros(max_iter)
    pl_errors_sum = np.zeros(max_iter)

    for _ in range(reps):
        ho_errors, pl_errors = test_convergence(leadership, max_iter)
        ho_errors_sum += ho_errors
        pl_errors_sum += pl_errors

    avg_ho_errors = ho_errors_sum / reps
    avg_pl_errors = pl_errors_sum / reps

    data = {
            'Iteration': np.arange(max_iter),
            'Avg_HO_Error': avg_ho_errors,
            'Avg_PL_Error': avg_pl_errors,
     }
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    return avg_ho_errors, avg_pl_errors


if __name__ == '__main__':

    out_file_7_1 = os.path.join(os.path.dirname(__file__), 'results', 'ex07.1.csv')
    average_convergence(reps=1000, leadership=False, output_file=out_file_7_1)

    out_file_7_2 = os.path.join(os.path.dirname(__file__), 'results', 'ex07.2.csv')
    average_convergence(reps=1000, leadership=True, output_file=out_file_7_2)






