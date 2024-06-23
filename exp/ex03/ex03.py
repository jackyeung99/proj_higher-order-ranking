import os
import sys
from sklearn.model_selection import ShuffleSplit

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.experiment_helpers import *


def run_experiment(dataset_file_path, splits, metric='std_likelihood'):

    for file in os.listdir(dataset_file_path):

        if file.endswith('.soc'):
            file_path = os.path.join(dataset_file_path, file)
            data, pi_values = read_strict_ordered_dataset(file_path)

            if len(data) > 50: 
                shuffle_split = ShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

                for idx, (train_index, test_index) in enumerate(shuffle_split.split(data)):

                    train_data = [data[i] for i in train_index]
                    test_data = [data[i] for i in test_index]

                    file_name = f'exp/ex03/data/f{file[:-4]}_split_{idx + 1}_results.csv'
                    run_models(train_data, test_data, pi_values, metric, file_name)
                
            


if __name__ == '__main__':
    # Run From ex03_realdata directory
    # rsync -zaP burrow:multi-reactive_rankings/higher_order_ranking/exp/ex03_realdata/data ~/senior_thesis/higher_order_ranking/exp/ex03_realdata/data

    dataset_file_path = 'datasets/processed_data'
    splits = 10
    run_experiment(dataset_file_path, splits) 