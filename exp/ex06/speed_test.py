import os
import sys
import time
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(repo_root)

from src import *
from exp.ex06.models.parallel_newman import *
from exp.ex06.models.numba_newman import *
from tst.tst_weight_conversion.old_newman import * 


def run_all_newman(weighted_data, pi_values):
    
    start = time.perf_counter() 
    compute_predicted_ratings_std(weighted_data, pi_values)
    compute_predicted_ratings_std_leadership(weighted_data, pi_values)
    compute_predicted_ratings_ho(weighted_data, pi_values)
    compute_predicted_ratings_hol(weighted_data, pi_values)
    end = time.perf_counter()
    tot_time  = end - start
    return tot_time

    
def run_all_newman_old(data, pi_values):
    start = time.perf_counter()
    compute_predicted_ratings_std_old(data, pi_values)
    compute_predicted_ratings_std_leadership_old(data, pi_values)
    compute_predicted_ratings_ho_old(data, pi_values)
    compute_predicted_ratings_hol_old(data, pi_values)
    end = time.perf_counter()
    tot_time = end - start

    return tot_time

def run_all_parrallel(weighted_data, pi_values):
    start = time.perf_counter()
    parrallel_std(weighted_data, pi_values)
    parallel_std_leadership(weighted_data, pi_values)
    parallel_ho(weighted_data, pi_values)
    parallel_hol(weighted_data, pi_values)
    end = time.perf_counter()
    tot_time = end - start

    return tot_time


def run_all_numba(weighted_data, pi_values): 

    start = time.perf_counter()
    numba_std(weighted_data, pi_values)
    numba_std_leadership(weighted_data, pi_values)
    numba_ho(weighted_data, pi_values)
    numba_hol(weighted_data, pi_values)
    end = time.perf_counter()
    tot_time = end - start

    return tot_time


def test_models(N, M, K1, K2):

    data, pi_values = generate_model_instance(N,M,K1,K2)
    weighted_data = convert_games_to_dict(data)

    std_model = run_all_newman(weighted_data, pi_values)
    old_newman = run_all_newman_old(data, pi_values)
    parallel = run_all_parrallel(weighted_data, pi_values)
    numba = run_all_numba(weighted_data, pi_values)

    return {'std_model': std_model, 'old_newman': old_newman, 'parrallel': parallel, 'numba': numba}


def test_M(results_dir):

    N=1000
    M_vec = np.logspace(2,6, num=10)
    K = 4

    results = []
    for m in M_vec:
        m = int(m)
        iteration = test_models(N, m, K, K)
        iteration['m'] = m
        results.append(iteration)

    
    df = pd.DataFrame(results)
    out_file = os.path.join(results_dir, 'ex06.1')
    df.to_csv(out_file, index=False)

def test_N(results_dir):

    N_vec= [10 ** i for i in range(1, 6)]
    M = 1000
    K = 4

    results = []
    for n in N_vec:
        n = int(n)
        iteration = test_models(n, M, K, K)
        iteration['n'] = n
        results.append(iteration)

    
    df = pd.DataFrame(results)
    out_file = os.path.join(results_dir, 'ex06.2')
    df.to_csv(out_file, index=False)

def test_K(results_dir):

    N = 1000
    M = 1000
    K_vec = [2, 8, 15, 30, 50]

    results = []
    for k in K_vec:
        iteration = test_models(N, M, k, k)
        iteration['k'] = k
        results.append(iteration)

    
    df = pd.DataFrame(results)
    out_file = os.path.join(results_dir, 'ex06.3')
    df.to_csv(out_file, index=False)

if __name__ == '__main__':

    results_path = os.path.join(repo_root, 'exp', 'ex06', 'results')
    os.makedirs(results_path, exist_ok=True)

    test_M(results_path)
    test_N(results_path)
    test_K(results_path)
