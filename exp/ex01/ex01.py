import os
import sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import traceback

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import generate_weighted_leadership_model_instance, generate_weighted_model_instance, split_weighted_dataset, run_models_synthetic

def process_rep(rep, N, M, K1, K2, file_dir, leadership):
    try:
        if leadership:
            data, pi_values = generate_weighted_leadership_model_instance(N, M, K1, K2)
        else:
            data, pi_values = generate_weighted_model_instance(N, M, K1, K2)
        

        for train_size in np.logspace(-2, 0, endpoint=False, num=25):
            training_set, testing_set = split_weighted_dataset(data, train_ratio=train_size)
            model_performance = run_models_synthetic(training_set, testing_set, pi_values)

            file_path = os.path.join(file_dir, f'train-{train_size}_rep-{rep+1}.csv')

            # Ensure the directory exists
            os.makedirs(file_dir, exist_ok=True)

            try:
                model_performance.to_csv(file_path)
                print(f"Saved file to {file_path}")
            except Exception as e:
                print(f"Error saving file: {e}")

    except Exception as e:
        print(f"Error in process_rep (rep={rep}): {e}")
        traceback.print_exc()

def evaluate_model_train_size(N, M, K1, K2, file_dir, executor, leadership=False):
    os.makedirs(file_dir, exist_ok=True)

    futures = []
    with ProcessPoolExecutor() as executor:
        for rep in range(250):
            futures.append(executor.submit(process_rep, rep, N, M, K1, K2, file_dir, leadership))

    for future in futures:
        try:
            future.result()  # This will raise any exceptions that were raised in the worker
        except Exception as e:
            print(f"Error in future: {e}")
            traceback.print_exc()




if __name__ == "__main__":

    base_path = os.path.dirname(__file__)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # standard 
        N, M, K1, K2 = 1000, 10000, 2, 2
        evaluate_model_train_size(N, M, K1, K2, os.path.join(base_path, 'data', 'ex01.1'), executor)

        # higher order 
        N, M, K1, K2 = 1000, 10000, 5, 5
        evaluate_model_train_size(N, M, K1, K2, os.path.join(base_path, 'data','ex01.2'), executor)

        # higher order leadership
        N, M, K1, K2 = 1000, 10000, 5, 5
        evaluate_model_train_size(N, M, K1, K2, os.path.join(base_path, 'data', 'ex01.3'), executor, leadership=True)
