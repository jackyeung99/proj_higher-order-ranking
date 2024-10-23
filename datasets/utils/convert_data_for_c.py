import os 
import sys


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from datasets.utils.extract_ordered_games import *
from datasets.utils.convert_raw_files import *
from src.utils.file_handlers import *
from src.utils.graph_tools import convert_dict_to_games


def group_datasets(file_directory):
    sub_files = {}
    for file in os.listdir(file_directory):
        dataset_id = file.split('_')[0]
        if dataset_id not in sub_files:
            sub_files[dataset_id] = []
        sub_files[dataset_id].append(file)
    return sub_files

def write_nodes(node_ids, out_file):
    with open(out_file, mode='w') as f:
        for node_id, node_name in node_ids.items():
            node_name = '_'.join(node_name.split())
            f.write(f"{node_id} {node_name}" + '\n')

def write_edges(games, out_file):
    with open(out_file, mode='w') as f:
        for game in games:
            line = ' '.join(map(str, game))
            f.write(line + '\n')

def read_edges(file_path):
    data, pi_values = read_edge_list(file_path)
    list_of_games = convert_dict_to_games(data)

    list_of_games = [tuple([x + 1 for x in game]) for game in list_of_games]
    return list_of_games

def read_nodes(file_path):
    name_conversions = get_alternative_names(file_path)
    # 1 index games
    name_conversions = {k+1: v for k,v in name_conversions.items()}
    return name_conversions

def loop_files(out_dir, data_dir, grouped_files):


    for dataset, files in grouped_files.items():
        
        files.sort()
        edges, nodes = files[0], files[1]

        edge_filepath = os.path.join(data_dir, edges)
        node_filepath = os.path.join(data_dir, nodes)
        
        games = read_edges(edge_filepath)
        node_ids = read_nodes(node_filepath)
      
        edge_out_file = os.path.join(out_dir, f'{dataset}_game.txt')
        node_out_file = os.path.join(out_dir, f'{dataset}_idx.txt')
        
        write_edges(games, edge_out_file)
        write_nodes(node_ids, node_out_file)



if __name__ == '__main__':


    data_dir = os.path.join(repo_root, 'datasets', 'processed_data')
    out_dir = os.path.join(repo_root, 'C_Prog', 'Readfile', 'Data')
    grouped_files = group_datasets(data_dir)

    loop_files(out_dir,data_dir, grouped_files)
