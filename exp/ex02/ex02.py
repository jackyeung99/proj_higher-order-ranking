import os
import sys
import logging
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *



def evaluate_models_fixed_train_size(N, M, K1, K2, file_dir, leadership=False, repetitions=1000, train_size=0.8):
    os.makedirs(file_dir, exist_ok=True)

    for rep in range(repetitions):
        if leadership:
            pi_values, data = generate_leadership_model_instance(N, M, K1, K2)
        else:
            pi_values, data = generate_model_instance(N, M, K1, K2)

        # Split data into training and testing sets
        training_set, testing_set = train_test_split(data, train_size=train_size, random_state=None)
        
        # Run models and save the results
        df = run_models(training_set, testing_set, pi_values, leadership)
        file_path = os.path.join(os.path.dirname(__file__), file_dir, f'rep-{rep+1}.csv')
        df.to_csv(file_path, index=False)
    

        
if __name__ == '__main__':

    base_path = os.path.dirname(__file__)
    # standard 
    N, M, K1, K2 = 5000, 10000, 2, 2
    evaluate_models_fixed_train_size(N, M, K1, K2, os.path.join(base_path, 'data','ex02.1'))

    # higher order 
    N, M, K1, K2 = 5000, 10000, 5, 5
    evaluate_models_fixed_train_size(N, M, K1, K2, os.path.join(base_path, 'data', 'ex02.2'))
    
    # higher order leadership
    N, M, K1, K2 = 5000, 10000, 5, 5
    evaluate_models_fixed_train_size(N, M, K1, K2, os.path.join(base_path, 'data', 'ex02.3'), leadership=True)
        