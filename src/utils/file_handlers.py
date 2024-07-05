
import os 
import sys
import csv
import pandas as pd
import re 

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils.operation_helpers import *

def save_results_to_csv(filename, headers, results):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data', filename)
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for result in results:
            writer.writerow(result)


def read_file_parameters(file):
    file_parameters = {}    
    file = file.replace('.csv', '')
    file_split = file.split('_')
    for param_value in file_split:
        param, value = param_value.split('-')
        file_parameters[param] = value

    return file_parameters

def read_dataset(file):

    file_split = file.split('_')
    dataset = file_split[0].replace('f', '')
    dataset = {'dataset': dataset}
    return dataset


def process_directory(compared_axis, base_path, directory, output_file, is_synthetic = True):
    os.makedirs(os.path.join(base_path, 'results'), exist_ok=True)

    results_proportions = []
    results_mean = []
    for file in os.listdir(os.path.join(base_path, 'data', directory)):
        if file.endswith('.csv'):
            file_path = os.path.join(base_path, 'data', directory, file)
            df = pd.read_csv(file_path)

            averages = calculate_column_means(df, compared_axis=compared_axis)
            proportions = calculate_percentages(df, compared_axis=compared_axis)

        
            if is_synthetic:
                file_info = read_file_parameters(file)
            else: 
                file_info = read_dataset(file)

            averages.update(file_info)
            proportions.update(file_info)

            results_mean.append(averages)
            results_proportions.append(proportions)
            
    
    mean_df = pd.DataFrame(results_mean).drop(columns=['Game'])
    proportion_df = pd.DataFrame(results_proportions).drop(columns=['Game'])

    mean_df.to_csv(os.path.join(base_path, 'results', f"{output_file}_means.csv"), index=False)
    proportion_df.to_csv(os.path.join(base_path, 'results', f"{output_file}_proportions.csv"), index=False)

def read_edge_list(file_path):
    data = {}

    with open(file_path) as file:
        for line in file.readlines():   
            if line.startswith('#'):
                split = line.split(':')
                pi_values = {player: 1.0 for player in range(split[1])}
            else:
                count, order = line.split(':')
                data[order] = count


    return data, pi_values




