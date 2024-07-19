import sys
import os
import numpy as np
import random
import pandas as pd
from scipy.stats import logistic, norm, uniform, expon, gamma, beta
from concurrent.futures import ProcessPoolExecutor
import traceback

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import normalize_scores, establish_order, run_models_synthetic, split_weighted_dataset

def generate_random_scores(N, distribution='logistic'):
    if distribution == 'logistic':
        return {n: float(np.exp(logistic.rvs(size=1)[0])) for n in range(N)}
    elif distribution == 'normal':
        return {n: float(np.exp(norm.rvs(size=1)[0])) for n in range(N)}
    elif distribution == 'uniform':
        return {n: float(np.exp(uniform.rvs(size=1)[0])) for n in range(N)}
    elif distribution == 'exponential':
        return {n: float(np.exp(expon.rvs(size=1)[0])) for n in range(N)}
    elif distribution == 'gamma':
        return {n: float(np.exp(gamma.rvs(a=1.0, size=1)[0])) for n in range(N)}
    elif distribution == 'beta':
        return {n: float(np.exp(beta.rvs(a=2.0, b=5.0, size=1)[0])) for n in range(N)}
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    

def generate_weighted_model_instance(N, M, K1, K2, distribution='logistic'):
    # Generate random scores based on specified distribution
    pi_values = generate_random_scores(N, distribution)
    normalize_scores(pi_values)

    list_of_nodes = list(range(N))
    data = {}

    for m in range(M):
        K = random.randint(K1, K2)
        tmp = random.sample(list_of_nodes, K)
        order = tuple(establish_order(tmp, pi_values))
        if order not in data:
            data[order] = 0
        data[order] += 1 

    return data, pi_values

def generate_weighted_leadership_model_instance(N, M, K1, K2, distribution='logistic'):
    # Generate random scores based on specified distribution
    pi_values = generate_random_scores(N, distribution)
    normalize_scores(pi_values)

    list_of_nodes = list(range(N))
    data = {}

    for m in range(M):
        K = random.randint(K1, K2)
        tmp = random.sample(list_of_nodes, K)
        order = establish_order(tmp, pi_values)
        
        f = order[0]
        order = order[1:]
        random.shuffle(order)
        order.insert(0, f)
        order = tuple(order)

        if order not in data:
            data[order] = 0
        data[order] += 1 
       
    return data, pi_values

def process_rep(data, pi_values, distribution, rep, file_dir):
    try:
        training_set, testing_set = split_weighted_dataset(data, train_ratio=.8)
        model_performance = run_models_synthetic(training_set, testing_set, pi_values)
        file_path = os.path.join(file_dir, f'distribution-{distribution}_rep-{rep+1}.csv')
        model_performance.to_csv(file_path)

    except Exception as e:
        print(f"Error in process_rep (rep={rep}): {e}")
        traceback.print_exc()



def run_experiments(N, M, K1, K2, leadership, repetitions, file_dir):
    os.makedirs(file_dir, exist_ok=True)

    N, M, K1, K2 = 1000, 5000, 4, 4
    distributions = ['logistic', 'normal', 'uniform', 'exponential', 'gamma', 'beta']
    for dist in distributions:
        if leadership:
            data, pi_values = generate_weighted_leadership_model_instance(N, M, K1, K2, distribution=dist)
        else:
            data, pi_values = generate_weighted_model_instance(N, M, K1, K2, distribution=dist)
            
        futures = []
        with ProcessPoolExecutor() as executor:
            for rep in range(repetitions):
                futures.append(executor.submit(process_rep, data, pi_values, dist, rep, file_dir))

    for future in futures:
        try:
            future.result()  # This will raise any exceptions that were raised in the worker
        except Exception as e:
            print(f"Error in future: {e}")
            traceback.print_exc()

            


if __name__ == '__main__':


    base_path = os.path.dirname(__file__)
    N, M, K1, K2 = 5000, 10000, 2, 2
    file_dir_6_1 = os.path.join(base_path, 'data', 'ex06.1')
    run_experiments(N, M, K1, K2, leadership=False, repetitions=100, file_dir=file_dir_6_1)

    N, M, K1, K2 = 5000, 10000, 5, 5
    file_dir_6_2 = os.path.join(base_path, 'data', 'ex06.2')
    run_experiments(N, M, K1, K2, leadership=False, repetitions=100, file_dir=file_dir_6_2)

    N, M, K1, K2 = 5000, 10000, 5, 5
    file_dir_6_3 = os.path.join(base_path, 'data', 'ex06.3')
    run_experiments(N, M, K1, K2, leadership=True, repetitions=100, file_dir=file_dir_6_2)
   