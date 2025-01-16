import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.file_handlers import group_dataset_files, read_dataset_files
from src.utils.operation_helpers import run_models



def evaluate_model_for_epoch(data, pi_values, dataset_id, epoch, train_size):
    """
    Evaluate the model for a given dataset and epoch. This function will be run in parallel.
    """
    train, test = train_test_split(data, train_size=train_size)
    results = run_models(train, test, pi_values)
    file_name = f"dataset-{dataset_id}_epoch-{epoch}.csv"
    results.to_csv(os.path.join(RESULTS_DIR, file_name))

def evaluate_models_fixed_train_size(epochs=20, train_size=0.8):
    grouped = group_dataset_files(DATA_DIR)
    for dataset in grouped:
        if int(dataset) not in [10, 11, 15, 41, 43, 44, 46, 47, 48, 49, 50, 51, 54, 55, 56, 58, 101]:
            futures = []
            with ProcessPoolExecutor() as executor:
                data, pi_values = read_dataset_files(grouped[dataset], DATA_DIR, is_synthetic=False)
                for epoch in range(epochs):
                    futures.append(executor.submit(evaluate_model_for_epoch, data, pi_values, dataset, epoch, train_size))
                
    # for dataset in grouped:
    #     if int(dataset) not in [10, 11, 15, 41, 43, 44, 46, 47, 48, 49, 50, 51, 54, 55, 56, 58, 101]:
    #         print(dataset)
    #         data, pi_values = read_dataset_files(grouped[dataset], DATA_DIR, is_synthetic=False)
    #         for epoch in range(epochs):
    #             evaluate_model_for_epoch(data, pi_values, dataset, epoch, train_size)


        
if __name__ == '__main__':
    DATA_DIR = os.path.join(repo_root, 'datasets', 'Real_Data')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')

    os.makedirs(RESULTS_DIR, exist_ok=True)

    evaluate_models_fixed_train_size()



        

