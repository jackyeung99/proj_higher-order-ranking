import re
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from datasets.utils.extract_ordered_games import *
from src.utils.file_handlers import *

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_alternative_names(filename):
    id_to_name = {}
    pattern = re.compile(r"# ALTERNATIVE NAME (\d+): (.+)")

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            match = pattern.match(line)
            if match:
                id = int(match.group(1))
                name = match.group(2)
                id_to_name[id] = name

    return id_to_name

def convert_names_to_id(games, name_to_id):
    ''' Given a dictionary of combined players from each file create new alternative names'''
    return {tuple(name_to_id[name] for name in ordered_tuple): count for ordered_tuple, count in games.items()}

def convert_id_to_name(games, id_to_name):
    ''' For each file convert the games to the names of the alternative names'''
    return {tuple(id_to_name[id] for id in ordered_tuple): count for ordered_tuple, count in games.items()}

def write_files(out_path, out_file, pi_values, games):
    ''' convert combined names back into ids and write the files'''
    alternative_names = list(pi_values.keys())
    name_to_id = {name: idx + 1 for idx, name in enumerate(alternative_names)}
    converted_games = convert_names_to_id(games, name_to_id)

    edge_file = os.path.join(out_path, f'{out_file}_edges.txt')
    write_edges(edge_file, converted_games, len(alternative_names))
    node_file = os.path.join(out_path, f'{out_file}_nodes.txt')
    write_nodes(node_file, name_to_id)

def write_edges(file, games, num_unique_players):
    with open(file, 'w') as file:
        file.write(f"# UNIQUE PLAYERS: {num_unique_players}\n")
        for ordered_tuple, count in sorted(games.items(), key = lambda x:x[1], reverse=True):
            votes_str = ','.join(map(str, ordered_tuple))
            file.write(f"{count}: {votes_str}\n")

def write_nodes(file, name_to_id):
    with open(file, 'w') as file:
        for name, id in name_to_id.items():
            file.write(f"# ALTERNATIVE NAME {id}: {name}\n")

def group_soi(file_directory):
    sub_files = {}
    for file in os.listdir(file_directory):
        if file and (file.endswith('.soc') or file.endswith('.soi')):
            split = file.replace('.soc', '').replace('.soi', '').split('-')
            dataset_id = split[0] 
            file_suffix = split[1]  
            
            if dataset_id not in sub_files:
                sub_files[dataset_id] = {}
            
            if file.endswith('.soi'):
                sub_files[dataset_id][file_suffix] = file
            elif file.endswith('.soc') and file_suffix not in sub_files[dataset_id]:
                sub_files[dataset_id][file_suffix] = file

    # Convert sub_files to a more usable format (list of files per dataset)
    grouped_files = {dataset_id: list(sub_data.values())  for dataset_id, sub_data in sub_files.items()}
    return grouped_files
     

def combine_soi(sub_files, file_directory, outfile):
    for dataset_id, files in sub_files.items():
        dataset_games = {}
        dataset_pi_values = {}

        for file in files:
            file_path = os.path.join(file_directory, file)
            id_to_name = get_alternative_names(file_path)
            data, pi_values = read_strict_ordered_dataset(file_path)
            converted_data = convert_id_to_name(data, id_to_name)
            
            for order, weight in converted_data.items():
                if order in dataset_games:
                    dataset_games[order] += weight
                else:
                    dataset_games[order] = weight

            for key in pi_values.keys():
                alt_name = id_to_name[key]
                dataset_pi_values[alt_name] = 1.0

        write_files(outfile, dataset_id, dataset_pi_values, dataset_games)

def convert_raw_files(file_path, read_function, title, outfile):
    dataset_games = {}
    dataset_pi_values = {}

    full_path = os.path.join(repo_root, 'datasets', 'raw_data', file_path)
    print(full_path)

    if os.path.isdir(full_path):
        for file in os.listdir(full_path):
            full_file_path = os.path.join(full_path, file)
            data, pi_values = read_function(full_file_path)

            for order, weight in data.items():
                if order in dataset_games:
                    dataset_games[order] += weight
                else:
                    dataset_games[order] = weight

            for key in pi_values.keys():
                dataset_pi_values[key] = 1.0
    else:
        data, pi_values = read_function(full_path)
        dataset_games = data
        dataset_pi_values = pi_values

    write_files(outfile, title, dataset_pi_values, dataset_games)

if __name__ == '__main__':
    file_directory = os.path.join(repo_root, 'datasets', 'raw_data', 'preflib')
    out_file_dir = os.path.join(repo_root, 'datasets', 'processed_data')

    grouping = group_soi(file_directory)
    combine_soi(grouping, file_directory, out_file_dir)

    convert_raw_files('authorships.txt', read_data_authors, '00104', out_file_dir)
    convert_raw_files('cl_data.txt', read_data_ucl, '00102', out_file_dir)
    convert_raw_files('fifa_wc.txt', read_data_fifa, '00103', out_file_dir)
    convert_raw_files('olympic_swimming', read_data_swimming, '00100', out_file_dir)
    convert_raw_files('horse_racing', read_data_horse, '00101', out_file_dir)

    