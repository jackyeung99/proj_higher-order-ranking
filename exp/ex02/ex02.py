import os
import sys
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.file_handlers import group_dataset_files
from src.utils.c_operation_helpers import run_simulation_synthetic


def evaluate_models_fixed_train_size(epochs=1000, train_size=0.8):
    grouped = group_dataset_files(DATA_DIR)

    for dataset in grouped:
        edge_file = grouped[dataset]['edges']
        node_file = grouped[dataset]['nodes']
        
        edge_path = os.path.join(DATA_DIR, edge_file)
        node_path = os.path.join(DATA_DIR, node_file)
        
        base_name = edge_file.replace('_edges.txt', '')
        for epoch in range(epochs):
            results = run_simulation_synthetic(node_path, edge_path, 1, train_size)
            file_name = f"{base_name}-epoch_{epoch}.csv"
            results.to_csv(os.path.join(RESULTS_DIR, file_name))


        
if __name__ == '__main__':
    DATA_DIR = os.path.join(repo_root, 'datasets', 'Synthetic_Data')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')

    os.makedirs(RESULTS_DIR, exist_ok=True)

    evaluate_models_fixed_train_size()



        

