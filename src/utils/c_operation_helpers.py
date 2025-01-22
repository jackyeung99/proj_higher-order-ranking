

import subprocess
import shlex
import numpy as np
import pandas as pd

import os 
import sys 

C_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'C_Prog'))
sys.path.append(C_PATH)

def run_simulation(filein_idx, filein_data, ratio=.8, is_synthetic=0):

    command = os.path.join(C_PATH, 'Readfile', 'bt_model_data.out') + ' ' + filein_idx + ' ' + filein_data + ' ' +  str(ratio) + ' ' + str(is_synthetic)
    # print(shlex.split(command))

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    
    ##parse output
    output = process.communicate()[0].decode("utf-8")
    # print(output)

    _, HO, HOL, BIN, BINL = output.split(';;;') 
    
    results = []
    for label, category_output in zip(['HO_BT', 'HOL_BT', 'BIN', 'BINL'], [HO, HOL, BIN, BINL]):
        category_output = category_output.split()
        results.append({
            'model': label,
            'av_error': category_output[0],
            'rho': category_output[1],
            'tau': category_output[2],
            'prior': category_output[3],
            'log-likelihood': category_output[4],
            'leadership-log-likelihood': category_output[5],
            'iterations': category_output[6]
        })

    return pd.DataFrame(results)


def split_output(convergence_result):
    
    data = [line.split() for line in convergence_result if line.strip()]

    # HOL_like, HO_like = data.pop(-1)

    data_np = np.array(data, dtype=float)
    std_convergence_criteria = data_np[:, 1]  
    log_connvergence_criteria = data_np[:, 2] 
    rms_convergence_criteria = data_np[:, 3] 

    return std_convergence_criteria, log_connvergence_criteria, rms_convergence_criteria

def run_simulation_convergence(filein_idx, filein_data, is_synthetic=0):

    
    command = os.path.join(C_PATH, 'Convergence_Readfile', 'bt_model_data.out') + ' ' + filein_idx + ' ' + filein_data + ' ' + str(is_synthetic)

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    
    ##parse output
    output = process.communicate()[0].decode("utf-8")
 
    HO, Z, BIN, BINZ = output.split(';;;')

    results = {}
    for label, category_output in zip(['HO', 'Z', 'BIN', 'BINZ'], [HO, Z, BIN, BINZ]):
        parsed_data = split_output(category_output.split('\t'))
        
        results[label] = {
            "std_convergence_criteria": parsed_data[0],
            "log_convergence_criteria": parsed_data[1],
            "rms_convergence_criteria": parsed_data[2],
        }
    
    
    
    return results 

