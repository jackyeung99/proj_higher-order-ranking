import os
import sys
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils import *

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def run_experiment(dataset_file_path, reps, file_name, leadership=False):

    os.makedirs(file_name, exist_ok=True)

    for file in os.listdir(dataset_file_path):

        if file.endswith('_edges.csv'):

            file_path = os.path.join(dataset_file_path, file)
            data, pi_values = read_edge_list(file_path)

            file_split = file.split('_')
            dataset_id = file_split[0]

            if len(data) > 50: 
        
                for rep in range(reps):

                    training_set, testing_set = train_test_split(data, train_size=.8, random_state=None)
                    out_file_name = os.path.join(repo_root, f'dataset-{dataset_id}_rep-{rep}.csv')
                    df = run_models(training_set, testing_set, pi_values, leadership=leadership)
                    df.to_csv(out_file_name, index=False)


                
            


if __name__ == '__main__':
    # Run From ex03_realdata directory
    # rsync -zaP burrow:multi-reactive_rankings/higher_order_ranking/exp/ex03/data ~/senior_thesis/higher_order_ranking/exp/ex03/data

    base_path = os.path.dirname(__file__)

    dataset_file_path = os.path.join('..', '..', 'datasets', 'processed_data')

    out_file = os.path.join(base_path, 'data', 'ex04.1')
    run_experiment(dataset_file_path, 1000, file_name=out_file) 

    out_file = os.path.join(base_path, 'data', 'ex04.2')
    run_experiment(dataset_file_path, 1000, file_name=out_file) 