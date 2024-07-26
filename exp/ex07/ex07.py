import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

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

def test_convergence( max_iter):

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

def save_convergence_data(rep, out_dir, max_iter):
    ho_errors, pl_errors = test_convergence(max_iter)
    
    data = {
        'Iteration': np.arange(max_iter),
        'Avg_HO_Error': ho_errors,
        'Avg_PL_Error': pl_errors,
    }

    df = pd.DataFrame(data)
    file_path = os.path.join(out_dir, f'rep-{rep}.csv')
    df.to_csv(file_path, index=False)

def average_convergence(reps, out_dir, max_iter=1000):
    os.makedirs(out_dir, exist_ok=True)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_convergence_data, rep, out_dir, max_iter) for rep in range(reps)]
        for future in futures:
            future.result()



if __name__ == '__main__':

    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    average_convergence(reps=1000, out_dir=out_dir)






