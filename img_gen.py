import numpy as np
import os
from PIL import Image
import pandas as pd
import math
import matplotlib.pyplot as plt

from dataset import convert_bitplane_to_image
from dataset import SEQUENCES, CLASSES

def read_pickled_dataframes():    
    data_dir = "./data" 
    dataframes = []
    for seq in SEQUENCES:
        files = os.listdir(f'{data_dir}/{seq}')
        for file in files:
            df = pd.read_pickle(f'{data_dir}/{seq}/{file}')
            df['filename'] = seq
            df['qpwz'] = file.split('_')[1]
            df['qp'] = file.split('_')[2].split('.')[0]
            dataframes.append(df)
    return dataframes


def gridify_df(df, lengths_grid, entropy_grid):
    df['length_grid'] = df['length'].apply(lambda x: lengths_grid[np.argmax(lengths_grid >= x)])
    df['entropy_grid'] = df['entropy'].apply(lambda x: entropy_grid[np.argmax(entropy_grid >= x)])
    return df


def filter_dfs(load=True):
    if load:
        dataframes = read_pickled_dataframes()
        df = pd.concat(dataframes)
    else:
        raise NotImplementedError
    lengths_grid = np.array(range(48, 1584 + 1, 24))
    entropy_grid = np.array([x / 100 for x in range(0 + 10, 100 + 1, 5)])
    df_filtered = pd.DataFrame()
    df_grid = gridify_df(df, lengths_grid, entropy_grid)
    for length in lengths_grid:
        for entropy in entropy_grid:
            filtered = df_grid[(df_grid['length_grid'] == length) & (df_grid['entropy_grid'] == entropy)]
            filtered = filtered.sample(min(len(filtered), 500))
            df_filtered = pd.concat([df_filtered, filtered], ignore_index=True)
    return df_filtered


if __name__ == "__main__":
    img_dir = "images"
    for c in CLASSES:
        os.makedirs(f'./{img_dir}/train/{c:04d}', exist_ok=True)
        os.makedirs(f'./{img_dir}/val/{c:04d}', exist_ok=True)
        os.makedirs(f'./{img_dir}/test/{c:04d}', exist_ok=True)
    print('Filtering dataframes...')
    filtered = filter_dfs()
    print('Generating images...')
    get_class = lambda row: CLASSES[np.argmax(CLASSES > row['length'])]
    for i, row in filtered.iterrows():
        convert_bitplane_to_image(row['data'], f'./{img_dir}/train/{get_class(row):04d}/{row["filename"]}_n{i}.png') # pyright: ignore 

