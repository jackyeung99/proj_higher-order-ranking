import pandas as pd 
import os


def loop_files():
    all_years = pd.DataFrame()
    raw_data_path = os.path.join(os.getcwd(), 'raw_data')
    for file in os.listdir(raw_data_path):
        print(file)
        file_path = os.path.join(raw_data_path, file)
        year_results = pd.read_csv(file_path)
        all_years = pd.concat([all_years, year_results], ignore_index=True)
    return all_years

def extract_edges(tournament_df):
    edges = []
    df = tournament_df
    for _ , grouping in df.groupby(['start_hole', 'teetime','round_num']):

        sorted_grouping = grouping.sort_values(by='round_score', ascending=True)
        edge = sorted_grouping['player_name'].to_list()
        edges.append(edge)
        
    return edges

def extract_pi_values(tournament_df):
    # Separate players who played fewer than three rounds
    player_rounds = tournament_df.groupby('player_name').size()
    players_cut = player_rounds[player_rounds < 3].index
    players_not_cut = player_rounds[player_rounds >= 3].index

    cut_players_df = tournament_df[tournament_df['player_name'].isin(players_cut)]
    not_cut_players_df = tournament_df[tournament_df['player_name'].isin(players_not_cut)]

    # order players who made the cut
    total_scores_not_cut = not_cut_players_df.groupby('player_name')['round_score'].sum().reset_index()
    total_scores_not_cut = total_scores_not_cut.sort_values(by='round_score', ascending=True)
    total_scores_not_cut['rank'] = total_scores_not_cut['round_score'].rank(method='min')

    # order cut players
    total_scores_cut = cut_players_df.groupby('player_name')['round_score'].sum().reset_index()
    total_scores_cut = total_scores_cut.sort_values(by='round_score', ascending=True)
    total_scores_cut['rank'] = total_scores_cut['round_score'].rank(method='min')

    combined_scores = pd.concat([total_scores_not_cut, total_scores_cut], ignore_index=True)
    ranking_dict = combined_scores.set_index('player_name')['rank'].to_dict()

    return ranking_dict
    

def extract_tournament(tournament_df):

    data = extract_edges(tournament_df)
    pi_values = extract_pi_values(tournament_df) 

    return data, pi_values

def extraction_workflow():
    all_golf_tournaments = loop_files()
    for _, tournament in all_golf_tournaments.groupby(['season', 'event_name']):
        data, pi_values = extract_tournament(tournament)
        print(data, pi_values)
        break


if __name__ == '__main__':
    extraction_workflow()