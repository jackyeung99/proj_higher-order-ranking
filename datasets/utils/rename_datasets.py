import os 
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

''' convert files '''

DATASET_NAMES = {
    # preflib data 
    '1': 'irish_election',
    '2': 'debian',
    '3': 'mariner_path',
    '4': 'netflix_prize',
    '5': 'burlington_election',
    '6': 'skate',
    '7': 'electoral_reform_society',
    '8': 'glasgow_city_council',
    '9': 'AGH_course_selection',
    '10': 'skiing',
    '11': 'web_search',
    '12': 't_shirt',
    '13': 'social_recommendation',
    '14': 'sushi',
    '15': 'clean_web_search',
    '16': 'aspen_election',
    '17': 'berkley_election',
    '18': 'minneapolis_election',
    '19': 'oakland_election',
    '20': 'pierce_election',
    '21': 'san_francisco_election',
    '22': 'san_leandro_election',
    '23': 'takoma_park_election',
    '24': 'mechanical_turk_dots',
    '25': 'mechanical_turk_puzzle',
    '26': '2002_french_presidential',
    '27': 'proto_french_election',
    '28': 'APA_election',
    '29': 'proto_french_election_ratings',
    '30': 'UK_labor_party_vote',
    '31': 'vermont_district_rates',
    '32': 'education_surveys_in_informatics',
    '33': 'san_sebastian_poster_competitions',
    '34': 'cities_survey',
    '35': 'breakfast_items',
    '36': 'kidney',
    '37': 'AAMAS_bidding',
    '38': 'project_bidding_data',
    '39': 'comp_sci_conference_bidding',
    '40': 'trip_advisor',
    '41': 'boardgames',
    '42': 'boxing',
    '43': 'cycling_races',
    '44': 'table_tennis_ranking',
    '45': 'tennis_ranking',
    '46': 'global_univeristy_ranking',
    '47': 'spotify_daily_chart',
    '48': 'spotify_countries_chart',
    '49': 'multilaps_competitions',
    '50': 'movehub_city_ranking',
    '51': 'countries_ranking',
    '52': 'formula_1_seasons',
    '53': 'formula_1_races',
    '54': 'weeks_power_ranking',
    '55': 'comibined_power_ranking',
    '56': 'seasons_power_ranking',
    '57': 'parliamentary_elections',
    '58': 'NSW_legislative_assembly_election',
    '59': 'campsongs',
    '60': 'polkadot',
    '61': 'Kusama',

    '99': 'nascar',
    # our data
    '100': 'olympic_swimming',
    '101': 'horse_racing',
    '102': 'UCL',
    '103': 'FIFA',
    '104': 'authorship',
    '105': 'wolf'
}

FLIPPED_DATASET_NAMES = {v:k for k,v in DATASET_NAMES.items()}




def rename_dataset_num_to_name(file_directory):
    base_path = os.path.join(repo_root, 'datasets', 'processed_data')

    for file in os.listdir(file_directory):

        file_split = file.split('-')

        try:
            dataset_num = str(int(file_split[0]))
            dataset_name = DATASET_NAMES[dataset_num]
            new_name = f'{dataset_name}-{file_split[1]}'
            os.rename(os.path.join(base_path, file), os.path.join(base_path, new_name))
        except:
            print('Dataset already renamed')
            
def rename_dataset_name_to_num(file_directory):
    base_path = os.path.join(repo_root, 'datasets', 'processed_data')

    for file in os.listdir(file_directory):
        file_split = file.split('-')
        try:
            datset_num = FLIPPED_DATASET_NAMES[file_split[0]]
            new_name = f'{datset_num.zfill(5)}-{file_split[1]}'

            os.rename(os.path.join(base_path, file), os.path.join(base_path, new_name))
        except:
            print('Dataset already renamed')



if __name__ == '__main__':


    rename_dataset_num_to_name('datasets/processed_data')
    rename_dataset_name_to_num('datasets/processed_data')