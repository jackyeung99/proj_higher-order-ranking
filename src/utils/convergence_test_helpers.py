import sys
import os 

import random 
import pandas as pd
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)


from src.models.zermello import compute_predicted_ratings_plackett_luce
from src.models.BradleyTerry import compute_predicted_ratings_HO_BT, MAX_ITER


# Run both models keeping track of error for iterations
def test_convergence(un_weighted_data, pi_values):
    _, HO_info = compute_predicted_ratings_HO_BT(un_weighted_data, pi_values, verbose=True)
    _, PL_info = compute_predicted_ratings_plackett_luce(un_weighted_data, pi_values, verbose=True)

    ho_errors = np.zeros(MAX_ITER)
    pl_errors = np.zeros(MAX_ITER)

    for i in range(1, MAX_ITER+1):
        ho_errors[i-1] = HO_info.get(i, 0)
        pl_errors[i-1] = PL_info.get(i, 0)

    return ho_errors, pl_errors

# for each repetition obtain error for each iteration
def save_convergence_data(file_name, data, pi_values):
    ho_errors, pl_errors = test_convergence(data, pi_values, MAX_ITER)
    
    data = {
        'Iteration': np.arange(MAX_ITER),
        'Avg_HO_Error': ho_errors,
        'Avg_PL_Error': pl_errors,
    }

    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)


