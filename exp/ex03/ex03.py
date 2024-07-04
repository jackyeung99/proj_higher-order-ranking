import os
import sys
import logging
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *

def evaluate_models_fixed_train_size(N, M, K1, K2, file_dir, leadership=False, repetitions=100, train_size=0.8):
   os.makedirs(file_dir, exist_ok=True)
    
   for rep in range(repetitions):
        if leadership:
            pi_values, data = generate_leadership_model_instance(N, M, K1, K2)
        else:
            pi_values, data = generate_model_instance(N, M, K1, K2)

         
        # Split data into training and testing sets
        training_set, testing_set = train_test_split(data, train_size=train_size, random_state=None)
        
        # Run models and save the results
        df = run_models(training_set, testing_set, pi_values)
        file_path = os.path.join(os.path.dirname(__file__), file_dir, f'rep_{rep+1}.csv')
        df.to_csv(file_path)
    

        
if __name__ == '__main__':
   M_values = [1000, 2500, 7500, 10000, 20000, 50000]
   for M in M_values: 
      N, K1, K2 = 5000, 4, 4
      evaluate_models_fixed_train_size(N, M, K1, K2, 'ex03.1')
      evaluate_models_fixed_train_size(N, M, K1, K2, 'ex03.2', leadership=True)

   K_values = [4, 8, 12, 16]
   for K in K_values:
      N, M = 5000, 5000
      evaluate_models_fixed_train_size(N, M, K, K, 'ex03.3')
      evaluate_models_fixed_train_size(N, M, K, K, 'ex03.4', leadership=True)
       
