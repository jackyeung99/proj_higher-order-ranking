import pandas as pd 
import os 

def loop_files():
    for file in os.listdir(os.getcwd()):
        if 'horse' in file:
            read_files(file)
            break


def read_files(file):
    df = pd.read_csv(file) 
    grouped_df = df.groupby('rid')


    with open('race_results.txt', 'a') as file:
        for race_id, group in grouped_df:
            group = group.sort_values(by='position')
            ordered_race = group['horseName'].to_list()

            hyperedge = '|'.join(map(str, ordered_race))
            file.write(f"{hyperedge}\n")
           

            

if __name__ == '__main__':
    loop_files()