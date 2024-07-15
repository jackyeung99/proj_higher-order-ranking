import os
import sys
from concurrent.futures import ProcessPoolExecutor


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *

def process_rep(rep, N, M, K1, K2, train_size, file_dir, file_name, leadership):
   if leadership:
      data, pi_values = generate_weighted_leadership_model_instance(N, M, K1, K2)
   else:
      data, pi_values = generate_weighted_model_instance(N, M, K1, K2)

   # Split data into training and testing sets
   training_set, testing_set = split_weighted_dataset(data, train_size=train_size, random_state=None)
   
   # Run models and save the results
   model_performance = run_models_synthetic(training_set, testing_set, pi_values)
   file_path = os.path.join(file_dir, f'{file_name}_rep-{rep+1}.csv')
   model_performance.to_csv(file_path, index=False)

def evaluate_models_fixed_train_size(N, M, K1, K2, file_dir, file_name, leadership=False, repetitions=100, train_size=0.8):
   os.makedirs(file_dir, exist_ok=True)

   futures  = []
   with ProcessPoolExecutor() as executor:
      for rep in range(repetitions):
         futures.append(executor.submit(process_rep, rep, N, M, K1, K2, train_size, file_dir, file_name, leadership))
   


        
if __name__ == '__main__':
   base_path = os.path.dirname(__file__)


   M_values = np.logspace(6, 18, num=10, endpoint=True, base=2.0)
   for M in M_values: 
      M = int(M)
      N, K1, K2 = 5000, 4, 4
      evaluate_models_fixed_train_size(N, M, K1, K2, os.path.join(base_path, 'data', 'ex03.1'), f'M-{M}')
      evaluate_models_fixed_train_size(N, M, K1, K2, os.path.join(base_path, 'data', 'ex03.2'), f'M-{M}', leadership=True)

   K_values = range(2,20,2)
   for K in K_values:
      N, M = 5000, 5000
      evaluate_models_fixed_train_size(N, M, K, K, os.path.join(base_path, 'data', 'ex03.3'), f'K-{K}')
      evaluate_models_fixed_train_size(N, M, K, K, os.path.join(base_path, 'data', 'ex03.4'), f'K-{K}',  leadership=True)
       
