
import os
import sys
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from datasets.utils.convert_raw_files import write_files
from src.utils.graph_tools import generate_model_instance, generate_leadership_model_instance

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def generate_synthetic_data(outfile_dir, epochs=50):
    N = 1000
    ratios = np.logspace(0, 2, num=20)
    K_values = range(2, 12, 2)
    leadership_options = [False, True]

    for is_leadership in leadership_options:
        for ratio in ratios:
            M = int(N * ratio)
            for K in K_values:
                for epoch in range(epochs):
                    if is_leadership:
                        data, pi_values = generate_leadership_model_instance(N, M, K, K)
                    else:
                        data, pi_values = generate_model_instance(N, M, K, K)

                    outfile = f'N-{N}_M-{M}_K-{K}_L-{int(is_leadership)}_epoch-{epoch}'
                    write_files(outfile_dir, outfile, pi_values, data)

                




if __name__ == '__main__':


    file_directory = os.path.join(repo_root, 'datasets', 'raw_data', 'preflib')
    out_file_dir = os.path.join(repo_root, 'datasets', 'Synthetic_Data')
    os.makedirs(out_file_dir, exist_ok=True)
    generate_synthetic_data(out_file_dir)

    # data, pi_values = generate_model_instance(1000, 1000, 5, 5)
    # outfile = f'N-1000_M-1000_K-5_L-0'
    # write_files(out_file_dir, outfile, pi_values, data)
