from collections import defaultdict
from datetime import date
import os 
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from datasets.utils.extract_ordered_games import *

''' functions that allow games and pi_values to be written in soi/soc format'''

def evaluate_uniform_edge_sizes(games):
    if not games:
        return True 
    
    first_length = len(games[0])
    for t in games:
        if len(t) != first_length:
            return False
    return True

def convert_names_to_id(games, alt_id):
    games_with_alt_id = []
    for game in games:
        new_game = tuple([alt_id[player] for player in game])
        games_with_alt_id.append(new_game)

    return games_with_alt_id

def convert_games_to_dict(games):
    # Count occurrences of each unique ordering
    unique_orderings = {}
    for game in games:
        ordering = tuple(game)  
        if ordering in unique_orderings:
            unique_orderings[ordering] += 1
        else: 
            unique_orderings[ordering] = 1
    
    return dict(sorted(unique_orderings.items(), key = lambda x:x[1],  reverse=True))

       

def to_soi(out_file, title, pi_values, games):

    alternative_names = list(pi_values.keys())
    name_to_id = {name: idx + 1 for idx, name in enumerate(alternative_names)}
    alt_id_games = convert_names_to_id(games, name_to_id)
    game_counts = convert_games_to_dict(alt_id_games)

    complete = evaluate_uniform_edge_sizes(games)
    if complete:
        file_name = os.path.join(repo_root,'datasets', 'processed_data',f'{out_file}.soc')
    else:
        file_name = os.path.join(repo_root,'datasets', 'processed_data',f'{out_file}.soi')
        
    with open(file_name, 'w') as file:
        # Write header information
        file.write(f"# TITLE: {title}\n")
        file.write(f"# PUBLICATION DATE: {date.today()}\n")
        file.write(f"# NUMBER ALTERNATIVES: {len(alternative_names)}\n")
        file.write(f"# NUMBER VOTERS: {len(games)}\n")
        file.write(f"# NUMBER UNIQUE ORDERS: {len(game_counts)}\n")
        
        # Write alternative names
        for idx, name in enumerate(alternative_names, start=1):
            file.write(f"# ALTERNATIVE NAME {idx}: {name}\n")
        
        # Write game data
        for ordered_tuple, count in game_counts.items():
            votes_str = ','.join(map(str, ordered_tuple))
            file.write(f"{count}: {votes_str}\n")



def convert_files(file_path, read_function, title): 
    full_path = os.path.join(repo_root, 'datasets', 'raw_data', file_path)

    print(full_path)

    if os.path.isdir(full_path):
        for idx, file in enumerate(os.listdir(full_path)):
            out_file = f'{title}-{str(idx + 1).zfill(9)}'
            file_full_path = os.path.join(full_path, file)
            data, pi_values = read_function(file_full_path)
            to_soi(out_file, title, pi_values, data)
    else: 
        out_file = f'{title}-000000001'
        data, pi_values = read_function(full_path)
        to_soi(out_file, title, pi_values, data)

if __name__ == '__main__':


    convert_files('authorships.txt', read_data_authors, 'authorship')
    convert_files('cl_data.txt', read_data_ucl, 'UCL')
    convert_files('fifa_wc.txt', read_data_fifa, 'FIFA')
    convert_files('olympic_swimming', read_data_swimming, 'olympic_swimming') 
    convert_files('horse_racing', read_data_horse, 'horse_racing')
    