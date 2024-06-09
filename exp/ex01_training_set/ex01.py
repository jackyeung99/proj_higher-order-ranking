import os
import sys
import csv


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.solvers import *
from exp.experiment_helpers.metrics import * 
from exp.experiment_helpers.model_evaluation import * 
from src.syntetic import *

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate_model_likelihood(N, M, K1, K2):

    for rep in range(50):
        for model in ['ho','hol']:
            if model == 'ho':
                experiment_train_size_ho(N, M, K1, K2, rep)
            else:
                experiment_train_size_hol(N, M, K1, K2, rep)
    

def experiment_train_size_ho(N, M, K1, K2, rep):

    pi_values, data = generate_model_instance(N, M, K1, K2)
    random.shuffle(data)
    for train_size in np.logspace(-2, 0, endpoint=False, num=25):
        training_set, testing_set = split_games(data, train_size)
        ho_likelihood, hol_likelihood, std_likelihood = benchmark_ho(training_set, testing_set, pi_values)

        file_name = f"N-{N}_M-{M}_K1-{K1}_K2-{K2}_train_size-{train_size}_rep-{rep}_model-ho.csv"
        save_instance_results(ho_likelihood, hol_likelihood, std_likelihood, FILE_DIR, file_name)
        
        
def experiment_train_size_hol(N, M, K1, K2, rep):

    pi_values, data = generate_leadership_model_instance(N, M, K1, K2)
    random.shuffle(data)
    for train_size in np.logspace(-2, 0, endpoint=False, num=25):
        training_set, testing_set = split_games(data, train_size)

        ho_likelihood, hol_likelihood, std_likelihood = benchmark_hol(training_set, testing_set, pi_values)

        file_name = f"N-{N}_M-{M}_K1-{K1}_K2-{K2}_train_size-{train_size}_rep-{rep}_model-hol.csv"
        save_instance_results(ho_likelihood, hol_likelihood, std_likelihood, FILE_DIR, file_name)



def run_experiments(M_values, K1_values, K2_values):
    # Evaluate for different M values
    for M in M_values:
        evaluate_model_likelihood(1000, M, K1=2, K2=6)
        
    # Evaluate for different K1 values
    for K1 in K1_values:
        K2 = K1 + 1
        evaluate_model_likelihood(1000, M=1500, K1=K1, K2=K2)
        
    # Evaluate for different K2 values
    for K2 in K2_values:
        evaluate_model_likelihood(N=1000, M=1500, K1=2, K2=K2)
        

if __name__ == '__main__':
    

    M_values = [500, 1000, 2000] 
    K1_values = [2, 4, 8, 16]
    K2_values = [2, 4, 8, 16] 

    run_experiments(M_values, K1_values, K2_values)