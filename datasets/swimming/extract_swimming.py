import pandas as pd


if __name__ == '__main__':

    df = pd.read_csv('datasets/swimming/raw_data/Olympic_Swimming_Results_1912to2020.csv').drop_duplicates()

    df= df[df['Results'] != 'Did not start']

    grouped = df.groupby(by=['Location', 'Year', 'Distance (in meters)', 'Stroke', 'Gender'])

    with open('race_results.txt', 'w') as file:
        for _, group in grouped:
            group[['Results']]
            group.sort_values(by='Results')
            ordered_race = group['Athlete'].to_list()

            hyperedge = '|'.join(map(str, ordered_race))
            file.write(f"{hyperedge}\n")


        