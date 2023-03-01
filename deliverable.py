# CSE 163 Project
# Author: Xingyuan Zhao, Mariana Li Chen, Wanjia Ruan

# Import Libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# Code
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Filter and rename the datasets
    data = df[df['year'].isin(range(2018, 2022))]
    data = data[['Country name', 'year', 'Life Ladder', 'Log GDP per capita',
                'Social support', 'Healthy life expectancy at birth',
                'Freedom to make life choices']]
    data.columns = ['country', 'year', 'score', 'GDP(log)', 'social',
                    'life expectancy', 'freedom']
    # Manipulate data
    data = data.dropna()
    data = data.astype({'score': float, 'GDP(log)': float, 'social': float, 
                        'life expectancy': float, 'freedom': float})
    return data


def join_data(data: pd.DataFrame, WORLD_FILE: str) -> gpd.GeoDataFrame:
    world = gpd.read_file(WORLD_FILE)
    world = world[['NAME', 'SUBUNIT', 'geometry']]
    # Combine world and happiness datasets
    world_data = data.merge(world, left_on='country', right_on='SUBUNIT', how='left')
    world_data = world_data.dropna()
    world_data_gpd = gpd.GeoDataFrame(world_data)
    return world_data_gpd


# Main
def main():
    print('Main Method')

    # Load data
    df = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/'
                       'cse-163-group-project/main/data/data.csv')
    data = clean_data(df)
    print('complete clean')

    # geodataframe
    WORLD_FILE = '/Users/rbc/Desktop/python/cse-163-group-project/data/world.shp'
    world_data = join_data(data, WORLD_FILE)
    print('complete join')


if __name__ == '__main__':
    main()