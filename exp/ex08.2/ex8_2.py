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
C_PATH = os.path.join(repo_root, 'C_Prog', 'Readfile')
sys.path.append(C_PATH)



def run_simulation_real_data (filein_idx, filein_data, model, ratio):
    
    print('----------------- Current Directory -----------------')
    print(os.getcwd())
    
    bt_model_data_path = os.path.join(C_PATH, 'bt_model_data.out')
    print(f"bt_model_data.out path: {bt_model_data_path}")
    print(f"filein_idx path: {filein_idx}")
    print(f"filein_data path: {filein_data}")
    # command = '../Readfile/bt_model_data.out ' + filein_idx + ' ' + filein_data + ' ' + str(model) + ' ' + str(ratio) 
    print('----------------- Command  -----------------')
    command = os.path.join(C_PATH, 'bt_model_data.out') + ' ' + filein_idx + ' ' + filein_data + ' ' + str(model) + ' ' + str(ratio)
    print(shlex.split(command))

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    

    ##parse output
    output, error = process.communicate()

    # Decode both output and error from bytes to string
    output = output.decode("utf-8")
    error = error.decode("utf-8")

    if process.returncode != 0:
        print(f"Subprocess failed with error: {error}")
        return None, None

    HO = {}
    HO['Iteration'] = int(output.split()[24])
    
    BIN = {}
    BIN['Iteration'] = int(output.split()[25])
    
    
    return HO, BIN 


def run_real_data(grouped_files, model, repetitions, out_file_dir):
    os.makedirs(out_file_dir, exist_ok=True)

    for dataset_id, dataset_files in grouped_files.items():
        print(dataset_id)
        dataset_files.sort()
        filein_data = os.path.join(C_PATH,'Data', dataset_files[0])
        filein_idx = os.path.join(C_PATH,'Data', dataset_files[1])
        

        data = []
        for rep in range(repetitions):   

            HO, BIN = run_simulation_real_data(filein_idx, filein_data, model, ratio=.8)

            iteration_result = {
                'Dataset': dataset_id,
                'HO_iterations': HO['Iteration'],
                'Model': model,
                'rep': rep 
            }

            data.append(iteration_result)

        pd.DataFrame(data).to_csv(os.path.join(out_file_dir, f'{dataset_id}_data.csv'))


def group_c_data(data_dir):
    sub_files = {}
    for file in os.listdir(data_dir):
        dataset_id = file.split('_')[0]
        if file not in ['cl_data_cprog_game.txt', 'cl_data_cprog_idx.txt', 'wc_data_cprog_game.txt', 'wc_data_cprog_idx.txt'] and int(dataset_id) not in [10, 11, 15, 41, 43, 44, 46, 47, 48, 49, 50, 51, 54, 55, 56, 58, 101]:
            if dataset_id not in sub_files:
                sub_files[dataset_id] = []
            sub_files[dataset_id].append(file)
    return sub_files



if __name__ == '__main__':
    dataset_file_path = os.path.join(C_PATH, 'Data')
    grouped = group_c_data(dataset_file_path)

    out_path = os.path.join(os.path.dirname(__file__), 'data')
    run_real_data(grouped, 1, 100, out_path)