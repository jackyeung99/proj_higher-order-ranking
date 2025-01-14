import os
import sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.file_handlers import group_dataset_files
from src.utils.c_operation_helpers import run_simulation


def evaluate_models_fixed_train_size(epochs=1000, train_size=0.8):
    grouped = group_dataset_files(DATA_DIR)

    for dataset in grouped:
        if dataset not in [10, 11, 15, 41, 43, 44, 46, 47, 48, 49, 50, 51, 54, 55, 56, 58, 101]:
            edge_file = grouped[dataset]['edges']
            node_file = grouped[dataset]['nodes']
            
            edge_path = os.path.join(DATA_DIR, edge_file)
            node_path = os.path.join(DATA_DIR, node_file)
            
            base_name = edge_file.replace('_edges.txt', '')
            for epoch in range(epochs):
                results = run_simulation(node_path, edge_path, train_size)
                file_name = f"{base_name}-epoch_{epoch}.csv"
                results.to_csv(os.path.join(RESULTS_DIR, file_name))


        
if __name__ == '__main__':
    DATA_DIR = os.path.join(repo_root, 'datasets', 'Real_Data')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')

    os.makedirs(RESULTS_DIR, exist_ok=True)

    evaluate_models_fixed_train_size()



        


# def process_rep(rep, data, pi_values, out_file_dir, dataset_id): 
#      training_set, testing_set = split_weighted_dataset(data)
#      model_performance = run_models(training_set, testing_set, pi_values)
#      file_path = os.path.join(out_file_dir, f'dataset-{dataset_id}_rep-{rep+1}.csv')
#      model_performance.to_csv(file_path, index=False)

# def process_file(file, dataset_file_path, repetitions, out_file_dir):
#     file_path = os.path.join(dataset_file_path, file)
#     data, pi_values = read_edge_list(file_path)
#     dataset_id = int(file.split('_')[0])
    
#     if dataset_id not in [10, 11, 15, 41, 43, 44, 46, 47, 48, 49, 50, 51, 54, 55, 56, 58, 101]:
#         print(file)
#         futures = []
#         with ProcessPoolExecutor(max_workers=32) as executor:
#             for rep in range(repetitions):
#                 futures.append(executor.submit(process_rep, rep, data, pi_values, out_file_dir, dataset_id))
               

# def run_experiment(dataset_file_path, repetitions, out_file_directory):
#     os.makedirs(out_file_directory, exist_ok=True)

#     for file in os.listdir(dataset_file_path):
#         if file.endswith('_edges.txt'):
#             process_file(file, dataset_file_path, repetitions, out_file_directory)

            


# if __name__ == '__main__':
#     # Run From ex03_realdata directory
#     # rsync -zaP burrow:multi-reactive_rankings/higher_order_ranking/exp/ex02/data ~/senior_thesis/higher_order_ranking/exp/ex02/

#     base_path = os.path.dirname(__file__)
#     dataset_file_path = os.path.join(repo_root, 'datasets', 'processed_data')
#     out_file = os.path.join(base_path, 'data')

#     run_experiment(dataset_file_path, 50, out_file_directory=out_file) 
