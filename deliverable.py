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
from sklearn.metrics import confusion_matrix,precision_score, recall_score,accuracy_score
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


def logistic_model_generate(data: pd.DataFrame):
    print("Training Logistic Model:")
    mean = data.score.mean()
    m2 = LogisticRegression(max_iter=2000)
    X = data[['CONTINENT', 'GDP_log', 'social', 'life_expectancy','freedom']]
    X = pd.get_dummies(X, columns = ["CONTINENT"],drop_first = True)
    y = data.score > mean
    m2.fit(X, y)
    print("Finished.")
    yhat = m2.predict(X) > 0.5
    confution_matrix  = confusion_matrix(y, yhat)
    accuracy = accuracy_score(y, yhat)
    precision = precision_score(y, yhat)
    recall = recall_score(y, yhat)
    f_score = 2/ ((1/precision) + (1/recall))
    print(
    "The confusion matrix is " , confution_matrix , " " ,
    "The accuracy is " , accuracy , "." ,
    "The precision is " , precision , "." ,
    "The recall is " , recall , "." ,
    "The F_score is " , f_score , "."
)
    return m2


def linear_model_generate(data: pd.DataFrame):
    print("Training linear Model:")
    m = smf.ols(
        "score ~ GDP_log + social + life_expectancy + freedom", data = data
    ).fit()
    return m



# Main
def main():
    print('Main Method')

    # Load data
    df = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/'
                       'cse-163-group-project/main/data/data.csv')
    data = clean_data(df)
    print('complete clean')

    # geodataframe
    WORLD_FILE = ('https://github.com/WanjiaRuan/cse-163-group-project/'
                  'blob/fd4af03bcaa683b7849afd75b7b6ee9acbfc0de9/data/world.zip?raw=true')
    world = gpd.read_file(WORLD_FILE)[['NAME', 'SUBUNIT', 'SUBREGION',
                                       'CONTINENT', 'POP_EST', 'geometry']]
    # join / merge dataset with geodataframe
    world_data = join_data(data, world)
    print('complete join')

    #map
    map_plot(world_data)
    print('complete plot')

    #Machine Learning Model
    print("ML Training")
    world_data_ml = world_data.rename(columns={"life expectancy": "life_expectancy", "GDP(log)": "GDP_log"})
    linear_model = linear_model_generate(world_data_ml)
    logistic_model = logistic_model_generate(world_data_ml)
    print(linear_model.summary())



if __name__ == '__main__':
    main()