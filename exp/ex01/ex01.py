import os
import sys
import csv
import logging
import concurrent.futures

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *



def evaluate_model_train_size(N, M, K1, K2, file_dir, executor, leadership=False):
    os.makedirs(file_dir, exist_ok=True)

    futures = []
    for rep in range(250):
        if leadership:
            data, pi_values = generate_leadership_model_instance(N, M, K1, K2)
        else:
            data, pi_values = generate_model_instance(N, M, K1, K2)
        random.shuffle(data)
        for train_size in np.logspace(-2, 0, endpoint=False, num=25):
            future = executor.submit(evaluate_single_instance, rep, train_size, data, pi_values, file_dir, leadership)
            futures.append(future)

    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print(result)

def evaluate_single_instance(rep, train_size, data, pi_values, file_dir, leadership):
    training_set, testing_set = split_games(data, train_size)
    # use leadership likelihood if testing leadership model
    df = run_models(training_set, testing_set, pi_values, leadership=leadership)
    file_path = os.path.join(os.path.dirname(__file__), file_dir, f'rep-{rep+1}_train-{train_size:.4f}.csv')
    df.to_csv(file_path, index=False)

if __name__ == "__main__":

    base_path = os.path.dirname(__file__)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # standard 
        N, M, K1, K2 = 5000, 10000, 2, 2
        evaluate_model_train_size(N, M, K1, K2, os.path.join(base_path, 'data', 'ex01.1'), executor)

        # higher order 
        N, M, K1, K2 = 5000, 10000, 5, 5
        evaluate_model_train_size(N, M, K1, K2, os.path.join(base_path, 'data','ex01.2'), executor)

        # higher order leadership
        N, M, K1, K2 = 5000, 10000, 5, 5
        evaluate_model_train_size(N, M, K1, K2, os.path.join(base_path, 'data', 'ex01.3'), executor, leadership=True)
