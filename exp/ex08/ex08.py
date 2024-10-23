import os
import sys

import random
import pandas as pd
import numpy as np 
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *
from src.utils.convergence_test_helpers import *


# for each file obtain data and run all repetitions 
def process_file(file, dataset_file_path, repetitions, out_file_dir):
    file_path = os.path.join(dataset_file_path, file)
    dataset_id = int(file.split('_')[0])

    # assume that read edge list returns weighted games
    data, pi_values = read_edge_list(file_path)

    # using un weighted data for Plackett Luce model 
    un_weighted_data = convert_dict_to_games(data)
    
    if dataset_id not in [10, 11, 15, 41, 43, 44, 46, 47, 48, 49, 50, 51, 54, 55, 56, 58, 101]:
        futures = []
        with ProcessPoolExecutor(max_workers=32) as executor:
            for rep in range(repetitions):
                #randomize the list of games
                random_data, _ = train_test_split(un_weighted_data, test_size=.2)
                file_name = os.path.join(out_file_dir, f"dataset-{dataset_id}_rep-{rep}.csv")
                futures.append(executor.submit(save_convergence_data, file_name, random_data, pi_values))





# loop through dataset file and process according
def run_experiment(dataset_file_path, repetitions, out_file_directory):
    os.makedirs(out_file_directory, exist_ok=True)

    for file in os.listdir(dataset_file_path):
        if file.endswith('_edges.txt'):
            process_file(file, dataset_file_path, repetitions, out_file_directory)



if __name__ == '__main__':

    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    base_path = os.path.dirname(__file__)
    dataset_file_path = os.path.join(repo_root, 'datasets', 'processed_data')

    run_experiment(dataset_file_path, 25, out_dir)

