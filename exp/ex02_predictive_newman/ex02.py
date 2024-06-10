import os
import sys
import csv
import logging

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.solvers import *
from src.syntetic import *
from exp.experiment_helpers.metrics import * 
from exp.experiment_helpers.model_evaluation import * 
from exp.experiment_helpers.file_handlers import * 

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate_model_prediction(N, M, K1, K2):

    for rep in range(50):

        for model in ['ho', 'hol']:

            if model == 'ho':
                ho_likelihood, hol_likelihood, std_likelihood = generate_and_benchmark_ho_model(N, M, K1, K2, train_size=.8)
            else: 
                ho_likelihood, hol_likelihood, std_likelihood = generate_and_benchmark_hol_model(N, M, K1, K2, train_size = .8)


            file_name = f"N-{N}_M-{M}_K1-{K1}_K2-{K2}_rep-{rep}_model-{model}.csv"
            save_instance_results(ho_likelihood, hol_likelihood, std_likelihood, FILE_DIR, file_name)


if __name__ == '__main__':

    # rsync -zaP burrow:multi-reactive_rankings/exp/ex02_predictive_newman/data ex02_data
    # parallel --jobs 24 python3 ex02.py {1} {2} {3} ::: 1000 ::: 64 141 312 689 1521 3360 7419 16384 ::: $(seq 2 10)
    N, M, K1 = map(int, sys.argv[1:])
    
    K2 = K1 + 1
    logging.debug(f'running code for {N} {M} {K1} {K2}')
    evaluate_model_prediction(N, M, K1, K2)
        