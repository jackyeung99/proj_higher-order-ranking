import os
import sys
from concurrent.futures import ProcessPoolExecutor


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils import *

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def process_rep(rep, data, pi_values, out_file_dir, dataset_id):
    training_set, testing_set = split_weighted_dataset(data)
    model_performance = run_models(training_set, testing_set, pi_values)
    file_path = os.path.join(out_file_dir, f'dataset-{dataset_id}_rep-{rep+1}.csv')
    model_performance.to_csv(file_path, index=False)

def process_file(file, dataset_file_path, repetitions, out_file_dir):
    file_path = os.path.join(dataset_file_path, file)
    data, pi_values = read_edge_list(file_path)
    dataset_id = file.split('_')[0]
    
    if all(len(x) > 1 for x in data.keys()) and len(data) > 50 and dataset_id not in [11, 41, 44, 46, 54, 55, 56]:
        futures = []
        with ProcessPoolExecutor() as executor:
            for rep in range(repetitions):
                futures.append(executor.submit(process_rep, rep, data, pi_values, out_file_dir, dataset_id))
               

def run_experiment(dataset_file_path, repetitions, out_file_directory):
    os.makedirs(out_file_directory, exist_ok=True)

    for file in os.listdir(dataset_file_path):
        if file.endswith('_edges.txt'):
            print(file)
            process_file(file, dataset_file_path, repetitions, out_file_directory)

            


if __name__ == '__main__':
    # Run From ex03_realdata directory
    # rsync -zaP burrow:multi-reactive_rankings/higher_order_ranking/exp/ex03/data ~/senior_thesis/higher_order_ranking/exp/ex03/data

    base_path = os.path.dirname(__file__)

    dataset_file_path = os.path.join(repo_root, 'datasets', 'processed_data')

    out_file = os.path.join(base_path, 'data', 'ex04.1')
    run_experiment(dataset_file_path, 20, file_name=out_file) 