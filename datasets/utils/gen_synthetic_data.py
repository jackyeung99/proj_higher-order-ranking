
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from datasets.utils.convert_raw_files import *
from src.utils.file_handlers import *
from src.utils.graph_tools import *

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def write_synthetic_files(out_path, out_file, pi_values, games):
    ''' 
    Function to call both edge and node write functions such that dataset is in a standard format 

    '''

    edge_file = os.path.join(out_path, f'{out_file}_edges.txt')
    write_edges(games, edge_file)
    node_file = os.path.join(out_path, f'{out_file}_nodes.txt')
    write_pi_values(pi_values, node_file)

def write_pi_values(node_ids, out_file):

    with open(out_file, mode='w') as f:
        for node_name, strength_score in node_ids.items():
            f.write(f"{node_name} {strength_score}" + '\n')

def generate_synthetic_data(outfile_dir):
    N = 1000
    ratios = np.logspace(0, 2, num=20)
    K_values = range(5, 50, 5)
    leadership_options = [True, False]

    for is_leadership in leadership_options:
        for ratio in ratios:
            M = int(N * ratio)
            for K in K_values:
                if is_leadership:
                    data, pi_values = generate_model_instance(N, M, K, K)
                else:
                    data, pi_values = generate_leadership_model_instance(N, M, K, K)


                outfile = f'N-{N}_M-{M}_K-{K}'
                write_synthetic_files(outfile_dir, outfile, pi_values, data)





if __name__ == '__main__':
    file_directory = os.path.join(repo_root, 'datasets', 'raw_data', 'preflib')
    out_file_dir = os.path.join(repo_root, 'datasets', 'Synthetic_Data')
    os.makedirs(out_file_dir, exist_ok=True)
    generate_synthetic_data(out_file_dir)

