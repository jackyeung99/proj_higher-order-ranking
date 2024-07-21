
import os 
import sys
import csv
import pandas as pd


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)


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

def process_file(file_path, metric):
    results = {} 
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        value = row[metric]
        model = row['model']
        
        results[model] = value

    return results
        
def process_directory(base_path, directory):
    os.makedirs(os.path.join(base_path, 'results', directory), exist_ok=True)

    results_log = []
    results_leadership = []
    results_rms = []
    results_rho = []
    results_tau = []

    for file in os.listdir(os.path.join(base_path, 'data', directory)):
        if file.endswith('.csv'):
            file_path = os.path.join(base_path, 'data', directory, file)
            file_info = read_file_parameters(file)

            # Process each metric and store the results
            results = process_file(file_path, 'log-likelihood')
            results.update(file_info)
            results_log.append(results)

            results = process_file(file_path, 'leadership-log-likelihood')
            results.update(file_info)
            results_leadership.append(results)

            results = process_file(file_path, 'rms')
            results.update(file_info)
            results_rms.append(results)

            results = process_file(file_path, 'rho')
            results.update(file_info)
            results_rho.append(results)

            results = process_file(file_path, 'tau')
            results.update(file_info)
            results_tau.append(results)

    # Create DataFrames for each metric
    log_likelihood_df = pd.DataFrame(results_log)
    leadership_log_likelihood_df = pd.DataFrame(results_leadership)
    rms_df = pd.DataFrame(results_rms)
    rho_df = pd.DataFrame(results_rho)
    tau_df = pd.DataFrame(results_rho)

    # Save each summary to a separate CSV file
    log_likelihood_df.to_csv(os.path.join(base_path, 'results', directory, 'log_likelihood_summary.csv'), index=False)
    leadership_log_likelihood_df.to_csv(os.path.join(base_path, 'results', directory, 'leadership_log_likelihood_summary.csv'), index=False)
    rms_df.to_csv(os.path.join(base_path, 'results', directory, 'rms_summary.csv'), index=False)
    rho_df.to_csv(os.path.join(base_path, 'results', directory, 'rho_summary.csv'), index=False)
    tau_df.to_csv(os.path.join(base_path, 'results', directory, 'tau_summary.csv'), index=False)

def process_directory_real_data(base_path):
    os.makedirs(os.path.join(base_path, 'results'), exist_ok=True)

    results_log = []
    results_leadership = []

    for file in os.listdir(os.path.join(base_path, 'data')):
        if file.endswith('.csv'):
            file_path = os.path.join(base_path, 'data', file)
            file_info = read_file_parameters(file)

            # Process each metric and store the results
            results = process_file(file_path, 'log-likelihoods')
            results.update(file_info)
            results_log.append(results)

            results = process_file(file_path, 'leadership-log-likelihood')
            results.update(file_info)
            results_leadership.append(results)


    # Create DataFrames for each metric
    log_likelihood_df = pd.DataFrame(results_log)
    leadership_log_likelihood_df = pd.DataFrame(results_leadership)

    # Save each summary to a separate CSV file
    log_likelihood_df.to_csv(os.path.join(base_path, 'results', 'log_likelihood_summary.csv'), index=False)
    leadership_log_likelihood_df.to_csv(os.path.join(base_path, 'results', 'leadership_log_likelihood_summary.csv'), index=False)



def read_edge_list(file_path):
    data = {}
    with open(file_path) as file:
        for line in file.readlines():   
            if line.startswith('#'):
                split = line.split(':')
                num = int(split[1].strip())
            else:
                count, order = line.split(':')
                count = int(count.strip())
                order = tuple(int(x) for x in order.split(','))
                if len(order) > 1:
                    data[order] = count


    pi_values = {player: 1.0 for player in range(num+1)}
    return data, pi_values




