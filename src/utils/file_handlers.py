
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
        

# needs refacoring 
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
    tau_df = pd.DataFrame(results_tau)

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



def read_node_list(file_path, is_synthetic=False):
    pi_values = {}
    with open(file_path) as f:
        for i in f.readlines():
            line = i.split()
            # set initial player rating to 1.0
            if is_synthetic: 
                pi_values[int(line[0])] = float(line[1])
            else:
                pi_values[int(line[0])] = 1.0  

    return pi_values


def read_edge_list(file_path):
    data = []
    with open(file_path) as f: 
        for line in f.readlines():
            # assuming first element is identifier of game size
            game = list(map(lambda x: int(x), line.split()[1:]))
            data.append(game)


    return data


def read_dataset_files(dataset_files: dict, file_directory, is_synthetic=False):
    if 'nodes' in dataset_files and 'edges' in dataset_files:
        edge_path = os.path.join(file_directory, dataset_files['edges'])
        data = read_edge_list(edge_path)
        node_path = os.path.join(file_directory, dataset_files['nodes'])
        pi_values = read_node_list(node_path, is_synthetic)

    return data, pi_values



def group_dataset_files(file_directory):
    sub_files = {}
    for file in os.listdir(file_directory):
        if file and (file.endswith('_nodes.txt') or file.endswith('_edges.txt')):
            dataset_id = file.replace('_nodes.txt', '').replace('_edges.txt', '')
            # dataset_id = ''.join(split[:-1]) 
        
            if dataset_id not in sub_files:
                sub_files[dataset_id] = {'nodes': None, 'edges': None}
            
            # Classify as nodes or edges
            if file.endswith('_nodes.txt'):
                sub_files[dataset_id]['nodes'] = file
            elif file.endswith('_edges.txt'):
                sub_files[dataset_id]['edges'] = file
            
        
    return sub_files




