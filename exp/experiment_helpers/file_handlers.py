
import os 
import csv
import pandas as pd 

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


def save_instance_results(ho_likelihood, hol_likelihood, std_likelihood, base_dir, file_name): 


    results_dir = os.path.join(base_dir, 'data')
    file_path = os.path.join(results_dir, file_name)

    os.makedirs(results_dir, exist_ok=True)

    with open(file_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["ho_likelihood", "hol_likelihood", "std_likelihood"])
        for ho, hol, std in zip(ho_likelihood, hol_likelihood, std_likelihood):
            writer.writerow([ho, hol, std])


def read_file_parameters(file):
    file_parameters = {}    
    file = file.replace('.csv', '')
    file_split = file.split('_')
    for param_value in file_split:
   
        param, value = param_value.split('-')
        file_parameters[param] = value

    return file_parameters

def calculate_percentages(df):
    total_rows = len(df)
    if total_rows == 0:
        return 0, 0

    prop_ho_greater_std = (df.iloc[:, 0] > df.iloc[:, 2]).sum() / total_rows
    prop_hol_greater_std = (df.iloc[:, 1] > df.iloc[:, 2]).sum() / total_rows

    return prop_ho_greater_std, prop_hol_greater_std

def process_data_directory(directory, output_file):
    results = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)

            prop_ho_greater_std, prop_hol_greater_std = calculate_percentages(df)
            file_parameters = read_file_parameters(file)

            result = {
                'prop_ho_greater_std': prop_ho_greater_std,
                'prop_hol_greater_std': prop_hol_greater_std
            }
            result.update(file_parameters)
            results.append(result)

    results_df = pd.DataFrame(results)
    
    results_df.to_csv(output_file, index=False)


