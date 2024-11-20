import os
import sys

import subprocess
import shlex

import random
import pandas as pd
import numpy as np 
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
C_PATH = os.path.join(repo_root, 'C_Prog')
# sys.path.append(C_PATH)
os.chdir(C_PATH)



def run_simulation_real_data (filein_idx, filein_data, model, ratio):
    
    
    

    command = 'Real_data/Convergence_Readfile/bt_model_data.out ' + filein_idx + ' ' + filein_data + ' ' + str(model) + ' ' + str(ratio) 
#     print(shlex.split(command))

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    
    ##parse output
    output = process.communicate()[0].decode("utf-8")
    R, RL, BIN, BINL = output.split(';;;')

    results = {}
    for label, category_output in zip(['R', 'RL', 'BIN', 'BINL'], [R, RL, BIN, BINL]):
        parsed_data = split_output(category_output.split('\t'))
        results[label] = {
            "std_convergence_criteria": parsed_data[0],
            "log_convergence_criteria": parsed_data[1],
            "rms_convergence_criteria": parsed_data[2],
        }
    
    
    
    return results 


def split_output(convergence_result):
    
    data = [line.split() for line in convergence_result if line.strip()]

    # HOL_like, HO_like = data.pop(-1)

    data_np = np.array(data, dtype=float)
    std_convergence_criteria = data_np[:, 1]  
    log_connvergence_criteria = data_np[:, 2] 
    rms_convergence_criteria = data_np[:, 3] 

    return len(std_convergence_criteria), len(log_connvergence_criteria), len(rms_convergence_criteria)



def run_real_data(grouped_files, model, repetitions, out_file_dir):
    os.makedirs(out_file_dir, exist_ok=True)

    for dataset_id, dataset_files in grouped_files.items():
        print(dataset_id)
        dataset_files.sort()
        filein_data = os.path.join(C_PATH,'Data', dataset_files[0])
        filein_idx = os.path.join(C_PATH,'Data', dataset_files[1])
        

        data = []
        for rep in range(repetitions):   

            convergence_dict = run_simulation_real_data(filein_idx, filein_data, model, ratio=.8)

            iteration_result = {
                'Dataset': dataset_id,
                'Ours': convergence_dict['R']['rms_convergence_criteria'],
                'Zermello': convergence_dict['RL']['rms_convergence_criteria'],
                'Ours_bin': convergence_dict['BIN']['rms_convergence_criteria'],
                'Zermello_bin': convergence_dict['BINL']['rms_convergence_criteria'],
                'criterion': 'rms_difference',
                'rep': rep 
            }

            data.append(iteration_result)

        pd.DataFrame(data).to_csv(os.path.join(out_file_dir, f'{dataset_id}_data.csv'))


def group_c_data(data_dir):
    sub_files = {}
    for file in os.listdir(data_dir):
        dataset_id = file.split('_')[0]
        if int(dataset_id) not in [1, 10, 11, 15, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 54, 55, 56, 58, 101, 105]:
            if dataset_id not in sub_files:
                sub_files[dataset_id] = []
            sub_files[dataset_id].append(file)
    return sub_files



if __name__ == '__main__':
    dataset_file_path = os.path.join(C_PATH, 'Data')
    grouped = group_c_data(dataset_file_path)

    out_path = os.path.join(os.path.dirname(__file__), 'data')
    run_real_data(grouped, 1, 20, out_path)