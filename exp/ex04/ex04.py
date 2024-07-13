import os
import sys
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils import *

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def split_weighted_dataset(dataset, train_ratio=0.8):
    # Step 1: Expand the dataset
    expanded_list = []
    for key, weight in dataset.items():
        expanded_list.extend([key] * weight)
    
    # Step 2: Shuffle the list
    random.shuffle(expanded_list)
    
    # Step 3: Split the list
    train_size = int(len(expanded_list) * train_ratio)
    train_list = expanded_list[:train_size]
    test_list = expanded_list[train_size:]
    
    # Step 4: Re-aggregate the list
    train_set = defaultdict(int)
    test_set = defaultdict(int)
    
    for item in train_list:
        train_set[item] += 1
        
    for item in test_list:
        test_set[item] += 1
    
    return dict(train_set), dict(test_set)

def run_experiment(dataset_file_path, reps, file_name, leadership=False):

    os.makedirs(file_name, exist_ok=True)

    for file in os.listdir(dataset_file_path):

        print(file)
        if file.endswith('_edges.txt'):

            file_path = os.path.join(dataset_file_path, file)
            data, pi_values = read_edge_list(file_path)
            
            if all(len(x) > 1 for x in data.keys()):
                file_split = file.split('_')
                dataset_id = file_split[0]

                if len(data) > 50: 
            
                    for rep in range(reps):
                        

                        training_set, testing_set = split_weighted_dataset(data)
                        out_file_name = os.path.join(repo_root, f'dataset-{dataset_id}_rep-{rep}.csv')
                        df = run_models(training_set, testing_set, pi_values, leadership=leadership)
                        df.to_csv(out_file_name, index=False)

 

                
            


if __name__ == '__main__':
    # Run From ex03_realdata directory
    # rsync -zaP burrow:multi-reactive_rankings/higher_order_ranking/exp/ex03/data ~/senior_thesis/higher_order_ranking/exp/ex03/data

    base_path = os.path.dirname(__file__)

    dataset_file_path = os.path.join(repo_root, 'datasets', 'processed_data')

    out_file = os.path.join(base_path, 'data', 'ex04.1')
    run_experiment(dataset_file_path, 1, file_name=out_file) 

    out_file = os.path.join(base_path, 'data', 'ex04.2')
    run_experiment(dataset_file_path, 1, file_name=out_file, leadership=True) 