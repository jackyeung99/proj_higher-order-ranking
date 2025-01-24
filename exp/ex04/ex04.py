import os
import sys

import pandas as pd


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.file_handlers import group_dataset_files
from src.utils.c_operation_helpers import run_simulation_convergence


def evaluate_convergence(epochs=50):
    grouped = group_dataset_files(DATA_DIR)

    for dataset in grouped:
        if int(dataset) not in [8, 9]:
            print(dataset)
        
            edge_file = grouped[dataset]['edges']
            node_file = grouped[dataset]['nodes']
                
            edge_path = os.path.join(DATA_DIR, edge_file)
            node_path = os.path.join(DATA_DIR, node_file)

            data = []
            for epoch in range(epochs):
                results = run_simulation_convergence(node_path, edge_path, is_synthetic=0)

                data.append({
                    'Dataset': dataset,
                    'Ours': len(results['HO']['rms_convergence_criteria']),
                    'Zermelo': len(results['Z']['rms_convergence_criteria']),
                    'Ours_bin': len(results['BIN']['rms_convergence_criteria']),
                    'Zermelo_bin': len(results['BINZ']['rms_convergence_criteria']),
                    'criterion': 'rms_difference',
                    'epoch': epoch 
                    })
                
                    

                pd.DataFrame(data).to_csv(os.path.join(RESULTS_DIR, f'{dataset}_data.csv'))

            


        
if __name__ == '__main__':
    DATA_DIR = os.path.join(repo_root, 'datasets', 'Real_Data')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    evaluate_convergence()



        

