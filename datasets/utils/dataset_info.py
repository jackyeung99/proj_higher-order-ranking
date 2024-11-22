import os 
import sys
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *
from datasets.utils.rename_datasets import DATASET_NAMES


def get_edge_size(data):

    edges = [len(x) for x in data.keys()]
    
    return min(edges), max(edges), np.mean(edges)



def create_info_table(dataset_file_directory, out_file): 


    dataset_info = []
    for file in os.listdir(dataset_file_directory):

        print(file)
        if file.endswith('_edges.txt'):

            file_path = os.path.join(dataset_file_directory , file)
            data, pi_values = read_edge_list(file_path)

            file_split = file.split('_')
            dataset_id = file_split[0]

            dataset_name = DATASET_NAMES[str(int(dataset_id))]
            N = len(pi_values)
            M = np.sum(list(data.values()))
            K1, K2, K_avg = get_edge_size(data)



            info = {
                    'dataset_id': dataset_id,
                    'name': dataset_name,
                    'N': N,
                    'M': M,
                    'Ratio': M/N,
                    'K1': K1,
                    'K2': K2,
                    'K_avg': round(K_avg, 3) 
                        }
            
            dataset_info.append(info)

    df = pd.DataFrame(dataset_info).sort_values(by=['dataset_id'])
    df.to_csv(out_file, index=False)




if __name__ == '__main__':

    dataset_dir = os.path.join(repo_root, 'datasets', 'processed_data')
    out_file = os.path.join(repo_root, 'datasets', 'dataset_info.csv')
    create_info_table(dataset_dir, out_file)

            
