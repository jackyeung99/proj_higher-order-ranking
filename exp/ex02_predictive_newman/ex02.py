import os
import sys
import csv

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

            
def run_experiments(N, M_values, K1_values): 
    for m in M_values:
        M = int(m)
        for K1 in K1_values:
            K2 = K1 + 1
            evaluate_model_prediction(N, M, K1, K2)
      

if __name__ == '__main__':

    N = 1000
    K1_values = [2, 4, 6, 8, 10, 12]
    M_values = np.logspace(6, 14, num=8, endpoint=True, base=2)
    
    run_experiments(N, M_values, K1_values)