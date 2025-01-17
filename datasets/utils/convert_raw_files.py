import re
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from datasets.utils.extract_ordered_games import read_data_authors, read_data_fifa, read_data_ucl, read_data_wolf, read_data_letor,  read_data_so
# from src.utils.file_handlers import *


def convert_names_to_id(games, name_to_id) -> dict:
    ''' Convert a list of ordered games such that each element in hyperedge is changed from id to names'''
    return [tuple(name_to_id[name] for name in ordered_tuple) for ordered_tuple in games]

def convert_id_to_name(games, id_to_name) -> dict:
    ''' Convert a list of ordered games such that each element in hyperedge is changed from id to names'''
    return [tuple(id_to_name[id] for id in ordered_tuple) for ordered_tuple in games]

def write_files(out_path, out_file, pi_values, games):
    ''' 
    Function to call both edge and node write functions such that dataset is in a standard format

    Args:
        out_path: The directory of the file where the files will be written
        out_file: The name of the dataset for which the edges and nodes are written 
        pi_values (dict): a dictionary of id: player for real data or player: player_rating for synthetic data 
        games (list(tuple)): a list of tuples of ordered interactions with result indicated by position, first is better then second, etc.  

    '''

    edge_file = os.path.join(out_path, f'{out_file}_edges.txt')
    write_edges(games, edge_file)
    node_file = os.path.join(out_path, f'{out_file}_nodes.txt')
    write_nodes(pi_values, node_file)


def write_nodes(node_ids, out_file): 
    '''
    Writes a dictionary of node IDs and names to a file in a standardized format.

    Args:
        node_ids (dict): A dictionary where keys are IDS and values are node names.
                         Example: {1: 'Node Name', 2:'Another Node'}.
        out_file (str): The name of the output file where the nodes will be written.

    Returns:
        None: This function writes the nodes to the specified file and does not return a value.

    Output:
        A file where each line is structured as:
        node_id node_name
        Node names are transformed to replace spaces with underscores.

    Example:
        Given `node_ids = {'Node A': 1, 'Node B': 2}` and `out_file = 'nodes.txt'`,
        the resulting file will contain:
        1 Node_A
        2 N
    '''


    with open(out_file, mode='w') as f:
        for node_id, node_name in node_ids.items():
            f.write(f"{node_id} {node_name}" + '\n')

def write_edges(games, out_file):
    '''
    Writes a list of games containing the interactions between nodes IDs in a standardized format.

    Args:
        games (list): A list of games in order of importance
        out_file (str): The name of the output file where the edges will be written.

    Returns:
        None: This function writes the edges to the specified file and does not return a value.

    Output:
        A file where each line is structured as:
        game length, interaction 


    Example:

        given games = [(1,2,3), (4,5,6)]
        The file will be:
        3 1 2 3 
        3 4 5 6 


    '''
    with open(out_file, mode='w') as f:
        for game in games:
            k = len(tuple(game))
            line = f'{k} '+ ' '.join(map(str, game))
            f.write(line + '\n')

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
    

def process_files(file_paths, read_function=None):
    """
    Process a list of file paths to aggregate games and pi_values.

    Args:
        file_paths (list of str): List of file paths to process.
        read_function (callable, optional): Function to read the dataset.
                                            Defaults to None, requiring direct file reading.

    Returns:
        tuple: A tuple containing:
            - dataset_games (list): List of game data tuples.
            - dataset_players (set): Set of player names or IDs.
    """
    dataset_games, dataset_players = [], set()

    for file_path in file_paths:
        if read_function:
            data, pi_values = read_function(file_path)
        else:
            data, pi_values = read_strict_ordered_dataset(file_path)
        
        # print(pi_values)
        dataset_games.extend(data)
        dataset_players.update(pi_values.keys())

    return dataset_games, dataset_players


def prepare_and_write(dataset_games, dataset_players, out_file, out_path):
    """
    Prepare the dataset and write edges and nodes to files.

    Args:
        dataset_games (list): List of game data tuples.
        dataset_players (set): Set of player names or IDs.
        out_file (str): Output file prefix.
        out_path (str): Directory to save the files.
    """

    dataset_pi_values = {i + 1: name for i, name in enumerate(dataset_players)}
    names_to_id = {name: idx for idx, name in dataset_pi_values.items()}
    converted_games = convert_names_to_id(dataset_games, names_to_id)

    write_files(out_path, out_file, dataset_pi_values, converted_games)


def combine_soi(sub_files, file_directory, outfile):
    """
    Combine grouped .soc and .soi files into datasets.

    Args:
        sub_files (dict): Mapping of dataset IDs to their respective files.
        file_directory (str): Directory containing the files.
        outfile (str): Directory to save the combined datasets.
    """
    allowed_datasets = {14, 45, 9, 28, 52, 44}

    for dataset_id, files in sub_files.items():
        if int(dataset_id) not in allowed_datasets:
            continue

        print(f"Processing dataset: {dataset_id}")
        file_paths = [os.path.join(file_directory, file) for file in files]
        dataset_games, dataset_pi_values = process_files(file_paths)

        # Special condition to exit early for certain datasets
        if dataset_id in {"9", "28"}:
            break

        prepare_and_write(dataset_games, dataset_pi_values, dataset_id, outfile)
  

def convert_raw_files(file_path, read_function, title, outfile):
    """
    Convert raw dataset files into a standardized format.

    Args:
        file_path (str): Path to the raw dataset directory or file.
        read_function (callable): Function to read the specific dataset.
        title (str): Title or ID for the output files.
        outfile (str): Directory to save the output files.
    """
    full_path = os.path.join(repo_root, "datasets", "Raw_Data", file_path)
    dataset_games, dataset_pi_values = [], set()

    print(f"Processing dataset: {title}")
    if os.path.isdir(full_path):
        file_paths = [os.path.join(full_path, file) for file in os.listdir(full_path)]
    else:
        file_paths = [full_path]

    dataset_games, dataset_pi_values = process_files(file_paths, read_function)
    prepare_and_write(dataset_games, dataset_pi_values, title, outfile)





if __name__ == '__main__':
    file_directory = os.path.join(repo_root, 'datasets', 'Raw_Data', 'preflib')
    out_file_dir = os.path.join(repo_root, 'datasets', 'Real_Data')
    os.makedirs(out_file_dir, exist_ok=True)


    # grouping = group_soi(file_directory)
    # combine_soi(grouping, file_directory, out_file_dir)


    # convert_raw_files('authorships.txt', read_data_authors, '00104', out_file_dir)


    convert_raw_files('fifa_wc.txt', read_data_fifa, '00001', out_file_dir)
    convert_raw_files('ucl_data.txt', read_data_ucl, '00002', out_file_dir)
    convert_raw_files('preflib/00014-00000001.soc', read_data_so, '00003', out_file_dir)
    convert_raw_files('preflib/00014-00000002.soi', read_data_so, '00004', out_file_dir)
    convert_raw_files('preflib/00009-00000002.soc', read_data_so, '00005', out_file_dir)
    convert_raw_files('preflib/00028-00000002.soi', read_data_so, '00006', out_file_dir)
    # network science 6
    convert_raw_files('letor/10032.soi', read_data_letor, '00008', out_file_dir)
    convert_raw_files('wolf.csv', read_data_wolf, '00009', out_file_dir)
    convert_raw_files('preflib/00047-00000001.soc', read_data_so, '00010', out_file_dir)

