import sys 
import os
import pandas as pd 
import re 

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from datasets.utils.extract_ordered_games import *
from src.utils.file_handlers import *

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

''' Functions to read files and extract games and pi_values'''
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

def read_data_horse(filename):
    df = pd.read_csv(filename)

    df['horseName'] = df['horseName'].apply(lambda x: '_'.join(str(x).split()))
    
    data = []
    pi_values = {k:1.0 for k in df['horseName']}

    grouped_df = df.groupby('rid')
    for race_id, group in grouped_df:
        group = group.sort_values(by='position')
        data.append(group['horseName'].to_list()) 

    # data = convert_games_to_dict(data)
    return data, pi_values

    

def read_data_swimming(filename):
    df = pd.read_csv(filename).drop_duplicates()

    df['Athlete'] = df['Athlete'].apply(lambda x: '_'.join(str(x).split()))
    df = df[(df['Results'] != 'Did not start') & (df['Relay?'] == False)]

    data = []
    pi_values = {k:1.0 for k in df['Athlete']}

    grouped = df.groupby(by=['Location', 'Year', 'Distance (in meters)', 'Stroke', 'Gender'])
    for _, group in grouped:
        group[['Results']]
        group.sort_values(by='Results')
        data.append(group['Athlete'].to_list())

    # data = convert_games_to_dict(data)
    return data, pi_values

def read_data_ucl (filename):

    data = []
    pi_values = {}

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            tmp = line.split('|')
            if len(tmp)>1:
                teams = tmp[2].split(',')
                game = []
                for i in teams:
                    player = '_'.join(i.split())  
                    pi_values[player] = 1.0
                    game.append(player)

                data.append(game)

    # data = convert_games_to_dict(data)
    return data, pi_values

def read_data_fifa (filename):

    data = []
    pi_values = {}

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            tmp = line.split('|')
            if len(tmp)>1:
                teams = tmp[1].split(',')
                game = []
                for i in teams:
                    team = i.strip().replace('West Germany', 'Germany').replace('East Germany', 'Germany')
                    i = '_'.join(team.split())  
                    pi_values[i] = 1.0
                    game.append(i)
                
                data.append(game)

    # data = convert_games_to_dict(data)
    return data, pi_values


def read_data_authors (filename):

    data = []
    pi_values = {}

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            tmp = line.split('|')
            if len(tmp)>1:
                teams = tmp[1].split(',')
                for i in teams:
                    pi_values[i] = 1.0
                data.append(teams)

    # data = convert_games_to_dict(data)
    return data, pi_values


def read_data_wolf(filename):
    data = []
    pi_values = {}

    with open(filename, 'r') as file:


        for line in file:
            i, j = line.strip().split(',')

            if i not in pi_values:
                pi_values[' '.join(i.strip().split())] = 1.0
            if j not in pi_values:
                pi_values[' '.join(j.strip().split())] = 1.0

            if i != j:
                data.append((i, j))

    # data = convert_games_to_dict(data)
    return data, pi_values

# Read PREFLIB Data  
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
                node_name = '_'.join(name.split())
                id_to_name[id] = node_name
 
    return id_to_name


def read_strict_ordered_dataset(filename):
    id_to_name = get_alternative_names(filename)

    data = []
    pi_values = {v:k for k, v in id_to_name.items()}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Extract votes and counts
            if line and not line.startswith("#"):
                try:
                    count, order = line.split(": ")
                    count = int(count)
                    order = tuple(map(lambda x: id_to_name[int(x)], order.split(",")))
                    data.extend([order] * count)
                    
                except:
                    print(filename)
     


    return data, pi_values