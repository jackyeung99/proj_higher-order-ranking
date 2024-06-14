import pandas as pd 


if __name__ == '__main__':

    drivers = pd.read_csv('drivers.csv')
    results = pd.read_csv('results.csv')
    races = pd.read_csv('races.csv')

    merged_table = pd.merge(drivers, results, on='driverId', how='left')
    merged_table = pd.merge(merged_table, races, on='raceId', how='left')

    merged_table['positionText'] = pd.to_numeric(merged_table['positionText'], errors='coerce')
    merged_table = merged_table.dropna(subset=['positionText'])
    merged_table['positionText'] = merged_table['positionText'].astype(int)
    grouped = merged_table.groupby('raceId')

    with open('race_results.txt', 'w') as file:

        for race_id, group in grouped:
            group = group.sort_values(by='positionText')
            ordered_race = group['driverId'].to_list()
               
            hyperedge = '|'.join(map(str, ordered_race))
            file.write(f"{hyperedge}\n")
            
