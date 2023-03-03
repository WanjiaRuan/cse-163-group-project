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
import plotly.express as px


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


def join_data(data: pd.DataFrame, world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Combine world and happiness datasets
    world_data = data.merge(world, left_on='country',
                            right_on='SUBUNIT', how='left')
    world_data = world_data.dropna()
    world_data_gpd = gpd.GeoDataFrame(world_data)
    return world_data_gpd


def map_plot(world_data: gpd.GeoDataFrame) -> None:
    yeardata = world_data.groupby('country')[['score', 'GDP(log)', 'social',
                                             'life expectancy',
                                             'freedom']].mean().copy()
    filter_world = world_data[['SUBUNIT', 'geometry']]
    yeardata = yeardata.merge(filter_world, left_on='country',
                              right_on='SUBUNIT', how='left')
    yeardata_gpd = gpd.GeoDataFrame(yeardata)
    yeardata_gpd.from_features(yeardata_gpd.set_index("SUBUNIT"), crs='WGS84')

    fig = px.choropleth_mapbox(yeardata_gpd, geojson=yeardata_gpd.geometry, 
                               locations=yeardata_gpd.index, 
                               color='score',color_continuous_scale="Viridis",
                               range_color=(2, 10),
                               center = {'lat': 47.65749, 'lon': -122.30385},
                               mapbox_style="carto-positron",
                               zoom=3, opacity=0.3,
                               hover_name='SUBUNIT',
                               labels={'score':'Average Score'},
                               title='Average Happiness Score Map (2018 - 2021)'
                               )
    
    fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0},
                      coloraxis_colorbar={'title':'Average Score (2018-2021) '})

    fig.show()


# Main
def main():
    print('Main Method')

    # Load data
    df = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/'
                       'cse-163-group-project/main/data/data.csv')
    data = clean_data(df)
    print('complete clean')

    # geodataframe
    WORLD_FILE = 'https://drive.google.com/file/d/1mGQaL7-HpLCZxPYRRYTniM2ZjYN80Sgl/view?usp=share_link'
    orld = gpd.read_file(WORLD_FILE)[['NAME', 'SUBUNIT', 'SUBREGION',
                                       'CONTINENT', 'POP_EST', 'geometry']]
    # join / merge dataset with geodataframe
    world_data = join_data(data, WORLD_FILE)
    print('complete join')

    #map
    map_plot(world_data)
    print('complete plot')


if __name__ == '__main__':
    main()