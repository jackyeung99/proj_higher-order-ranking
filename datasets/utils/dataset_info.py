import os 
import sys
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src import *
from datasets.utils.rename_datasets import DATASET_NAMES


def get_edge_size(data):
    edges = [len(x) for x in data]
    
    return min(edges), max(edges), np.mean(edges)



def create_info_table(DATA_DIR, out_file): 

    grouped = group_dataset_files(DATA_DIR)

    dataset_info = []
    for dataset in grouped:

        data, pi_values = read_dataset_files(grouped[dataset], DATA_DIR)

        dataset_name = DATASET_NAMES[str(int(dataset))]
        edge_len = [len(x) for x in data]
        info = {
                'Dataset_ID': dataset,
                'Name': dataset_name,
                'N': len(pi_values),
                'M': len(data),
                'R': np.round(len(data)/len(pi_values),3),
                'K1': min(edge_len),
                'K2': max(edge_len),
                'K_avg': np.round(np.average(edge_len), 3) 
                    }
        
        dataset_info.append(info)

    df = pd.DataFrame(dataset_info).sort_values(by=['Dataset_ID'])
    df.to_csv(out_file, index=False)




if __name__ == '__main__':

    DATA_DIR = os.path.join(repo_root, 'datasets', 'Real_Data')
    out_file = os.path.join(repo_root, 'datasets', 'dataset_info.csv')
    create_info_table(DATA_DIR, out_file)

            
