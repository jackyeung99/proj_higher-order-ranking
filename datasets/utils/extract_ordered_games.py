import sys 
import os
import pandas as pd 

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

    data = []
    pi_values = {k:1.0 for k in df['horseName']}

    grouped_df = df.groupby('rid')
    for race_id, group in grouped_df:
        group = group.sort_values(by='position')
        data.append(group['horseName'].to_list()) 

    data = convert_games_to_dict(data)
    return data, pi_values

    

def read_data_swimming(filename):
    df = pd.read_csv(filename).drop_duplicates()
    df = df[(df['Results'] != 'Did not start') & (df['Relay?'] == False)]

    data = []
    pi_values = {k:1.0 for k in df['Athlete']}

    grouped = df.groupby(by=['Location', 'Year', 'Distance (in meters)', 'Stroke', 'Gender'])
    for _, group in grouped:
        group[['Results']]
        group.sort_values(by='Results')
        data.append(group['Athlete'].to_list())

    data = convert_games_to_dict(data)
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
                for i in range(0, len(teams)):
                    pi_values[teams[i]] = 1.0
                data.append(teams)

    data = convert_games_to_dict(data)
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
                for i in range(0, len(teams)):
                    teams[i] = teams[i].replace('West Germany', 'Germany')
                    teams[i] = teams[i].replace('East Germany', 'Germany')
                    pi_values[teams[i]] = 1.0
                data.append(teams)

    data = convert_games_to_dict(data)
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
                for i in range(0, len(teams)):
                    pi_values[teams[i]] = 1.0
                data.append(teams)

    data = convert_games_to_dict(data)
    return data, pi_values


def read_strict_ordered_dataset(filename):
    data = {}
    pi_values = {}

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Extract votes and counts
            if line and not line.startswith("#"):
                try:
                    count, order = line.split(": ")
                    count = int(count)
                    # 0 index real data
                    order = tuple(map(lambda x: int(x) - 1, order.split(",")))

                    data[order] = count
                    # add all unique players to pi_values
                    for id in order:
                        if id not in pi_values:
                            pi_values[id] = 1.0
                except:
                    print(filename)
     


    return data, pi_values
