import os
import sys
import csv
import logging

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *



def evaluate_model_likelihood(N, M, K1, K2):

    for rep in range(20):
        pi_values, data = generate_model_instance(N, M, K1, K2)
        random.shuffle(data)
        for train_size in np.logspace(-2, 0, endpoint=False, num=25):
            training_set, testing_set = split_games(data, train_size)
    
            file_name = os.path.join(repo_root, f"exp/ex01/data/N-{N}_M-{M}_K1-{K1}_K2-{K2}_trainsize-{train_size}_rep-{rep}.csv")
            df = run_models(training_set, testing_set, pi_values, file_name)
            print(calculate_percentages(df))
            




        


if __name__ == "__main__":
# rsync -zaP burrow:multi-reactive_rankings/higher_order_ranking/exp/ex01_training_size/data ~/senior_thesis/higher_order_ranking/exp/ex01_training_size/

# parallel --jobs 24 python3 ex01.py {1} {2} {3} {4} ::: 1000 ::: 1000 1500 2000 ::: $(seq 2 10) ::: $(seq 2 10)
    N, M, K1, K2 = map(int, sys.argv[1:])
    if int(K2) < int(K1):
        exit()
    else:
        logging.debug(f'running code for {N} {M} {K1} {K2}')
        evaluate_model_likelihood(N, M, K1, K2)
        



