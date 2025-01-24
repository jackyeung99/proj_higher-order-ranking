import os
import sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.file_handlers import group_dataset_files, read_file_parameters
from src.utils.c_operation_helpers import run_simulation_convergence


def evaluate_convergence(buffer_size=25):

    grouped = group_dataset_files(DATA_DIR)
    output_file = os.path.join(RESULTS_DIR, 'Convergence_Table.csv')

    # Check if file exists for header management
    headers = ['N', 'M', 'K', 'L', 'epoch', 'Ours', 'Zermelo', 'Ours_bin', 'Zermelo_bin', 'criterion']
    pd.DataFrame(columns=headers).to_csv(output_file, index=False)

    buffer = []
    for dataset in grouped:

        edge_file = grouped[dataset]['edges']
        node_file = grouped[dataset]['nodes']

        edge_path = os.path.join(DATA_DIR, edge_file)
        node_path = os.path.join(DATA_DIR, node_file)

        results = run_simulation_convergence(node_path, edge_path, is_synthetic=1)

        epoch = read_file_parameters(edge_file.replace('_edges.txt', ''))
        epoch.update({
            'Ours': len(results['HO']['rms_convergence_criteria']),
            'Zermelo': len(results['Z']['rms_convergence_criteria']),
            'Ours_bin': len(results['BIN']['rms_convergence_criteria']),
            'Zermelo_bin': len(results['BINZ']['rms_convergence_criteria']),
            'criterion': 'rms_difference',
        })
        print(read_file_parameters(edge_file.replace('_edges.txt', '')))

        buffer.append(epoch)

        if len(buffer) >= buffer_size:
            pd.DataFrame(buffer).to_csv(output_file, mode='a', index=False, header=False)
            file_exists = True  # Headers have now been written
            buffer = []

    # Write any remaining data in the buffer
    if buffer:
        pd.DataFrame(buffer).to_csv(output_file, mode='a', index=False, header=False)


            


        
if __name__ == '__main__':
    DATA_DIR = os.path.join(repo_root, 'datasets', 'Synthetic_Data')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    evaluate_convergence()



        

