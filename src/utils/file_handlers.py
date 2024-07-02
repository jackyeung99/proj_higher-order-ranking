
import os 
import csv
import pandas as pd
import re 

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

def calculate_percentages(df, compared_axis):
    total_rows = len(df)
    if total_rows == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    comparisons = [df.iloc[:, col] > df.iloc[:, compared_axis] for col in range(1,10) if col != compared_axis]
    percentages = [comparison.sum() / total_rows for comparison in comparisons]

    return tuple(percentages)

def calculate_column_means(df):
    if df.empty:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    means = [df.iloc[:, col].mean() for col in range(1,10)]

    return tuple(means)

def process_directory_prop(directory, output_file, is_synthetic=True):
    results = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)


            proportions = calculate_percentages(df)
            if is_synthetic:
                file_info = read_file_parameters(file)
            
            else: 
                file_info = read_dataset(file)

            result = {
                'prop_stdl_>_std': proportions[0],
                'prop_ho_>_std': proportions[1],
                'prop_hol_>_std': proportions[2],
                'prop_point_>_std': proportions[3],
                'prop_spring_>_std': proportions[4]
            }
            result.update(file_info)
            results.append(result)

    results_df = pd.DataFrame(results)
    
    results_df.to_csv(output_file, index=False)


def process_directory_mean(directory, output_file, is_synthetic = True):
    results = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            averages = calculate_column_means(df)
        
            if is_synthetic:
                file_info = read_file_parameters(file)
            else: 
                file_info = read_dataset(file)

            result = {
                'std': averages[0],
                'stdl': averages[1],
                'ho': averages[2],
                'hol': averages[3],
                'point': averages[4],
                'spring_rank': averages[5],
            }
            result.update(file_info)
            results.append(result)

    results_df = pd.DataFrame(results)
    
    results_df.to_csv(output_file, index=False)




