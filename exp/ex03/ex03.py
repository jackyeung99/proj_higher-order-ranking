import os 
import sys

import pandas as pd 
import numpy as np

import subprocess
import shlex


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
C_PATH = os.path.join(repo_root, 'C_Prog')
# sys.path.append(C_PATH)
os.chdir(C_PATH)



def run_simulation (N, M, K1, K2, ratio, model):
    
    
    

    command = 'Synthetic/bt_model.out ' + str(N) + ' ' + str(M) + ' ' + str(K1) + ' ' + str(K2) + ' ' + str(model) + ' ' + str(ratio) 
#     print(shlex.split(command))

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    
    
    ##parse output
    output = process.communicate()[0].decode("utf-8")
#     print (output)


    G = {}
    G['N'] = int(output.split()[0])
    G['M'] = int(output.split()[1])
    G['prior'] = float(output.split()[2])
    G['like_ho'] = float(output.split()[3])
    G['like_hol'] = float(output.split()[4])
    G['like_bin'] = float(output.split()[5])

    
    HO = {}
    HO['log_err'] = float(output.split()[6])
    HO['spear'] = float(output.split()[7])
    HO['kend'] = float(output.split()[8])
    HO['prior'] = float(output.split()[9])
    HO['like_ho'] = float(output.split()[10])
    HO['like_hol'] = float(output.split()[11])
    HO['Iteration'] = int(output.split()[30])
    
    HOL = {}
    HOL['log_err'] = float(output.split()[12])
    HOL['spear'] = float(output.split()[13])
    HOL['kend'] = float(output.split()[14])
    HOL['prior'] = float(output.split()[15])
    HOL['like_ho'] = float(output.split()[16])
    HOL['like_hol'] = float(output.split()[17])
    HOL['Iteration'] = int(output.split()[31])
    
    BIN = {}
    BIN['log_err'] = float(output.split()[18])
    BIN['spear'] = float(output.split()[19])
    BIN['kend'] = float(output.split()[20])
    BIN['prior'] = float(output.split()[21])
    BIN['like_ho'] = float(output.split()[22])
    BIN['like_hol'] = float(output.split()[23])

    Z = {}
    Z['log_err'] = float(output.split()[24])
    Z['spear'] = float(output.split()[25])
    Z['kend'] = float(output.split()[26])
    Z['prior'] = float(output.split()[27])
    Z['like_ho'] = float(output.split()[28])
    Z['like_hol'] = float(output.split()[29])
    Z['Iteration'] = int(output.split()[32])
    
    return G, HO, HOL, BIN, Z


def hyper_edge_iteration(repetitions, out_file_dir, K_values):
    os.makedirs(out_file_dir, exist_ok=True)
    n = 10
    values = []
    for r in ratios:
        print(r)
        for rep in range(repetitions):
            m = int(r * n)
            _, HO, _, _, Z = run_simulation(n, m, 5, 5, 1.0, 1)
            values.append({
                "Ratio": r,
                "Rep": rep,
                "Ours": HO['Iteration'],
                "Outs_Log_like": HO['log_err'],
                "Zermello": Z['Iteration'],
                "Zermello_log_like": Z['log_err']
                }) 

    df = pd.DataFrame(values)
    path = os.path.join(out_file_dir, 'ratio_iteration_count.csv')
    df.to_csv(path)


if __name__ == '__main__':
    ratios = np.logspace(0, 2, num=20)
    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    reps = 1000
    hyper_edge_iteration(reps, out_dir, ratios)
