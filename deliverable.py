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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, \
                            recall_score, accuracy_score, f1_score, \
                            mean_squared_error
import plotly.express as px


# Code
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Filter and rename the datasets
    data = df[df['year'].isin(range(2018, 2022))]
    data = data[['Country name', 'year', 'Life Ladder', 'Log GDP per capita',
                 'Social support', 'Healthy life expectancy at birth',
                 'Freedom to make life choices']]
    data.columns = ['country', 'year', 'score', 'GDP_log', 'social',
                    'life_expectancy', 'freedom']
    # Manipulate data
    data = data.dropna()
    data = data.astype({'score': float, 'GDP_log': float, 'social': float,
                        'life_expectancy': float, 'freedom': float})
    return data


def join_data(data: pd.DataFrame,
              world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Combine world and happiness datasets
    world_data = data.merge(world, left_on='country',
                            right_on='SUBUNIT', how='left')
    world_data = world_data.dropna()
    world_data_gpd = gpd.GeoDataFrame(world_data)
    return world_data_gpd


def yeardata(data: pd.DataFrame) -> dict[list]:
    res = {}
    for y in range(2018, 2022):
        store = {}
        year = data[data['year'] == y]
        c = pd.cut(year["score"], [2 + 0.25 * x for x in range(25)])
        for i, v in c.value_counts().to_dict().items():
            store[i.left] = v
        x = sorted(store.items())
        res[y] = [i[1] for i in x]
    return res


def score_distr(df: pd.DataFrame) -> None:
    nice_color = sns.color_palette("BuPu_r", 4)
    score_distr_graph = sns.displot(data=df, x="score", bins=24,
                                    height=6, aspect=1.4, hue="year",
                                    palette=nice_color, kde=True)
    score_distr_graph.set(xlabel="Happiness Score",
                          title="Happiness Score Distribution Histogram")

    plt.savefig('scoredistribution.png')


def score_plot(df: pd.DataFrame) -> None:
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(17, 17))
    score_vs_gdp_graph = sns.scatterplot(x='GDP_log', y='score',
                                         hue="year", data=df, ax=ax1)
    score_vs_gdp_graph.set(xlabel="Happiness Score",
                           ylabel="Log GDP per Capita",
                           title="Happiness Score vs Log GDP per Capita")
    score_vs_social_graph = sns.scatterplot(x='social', y='score',
                                            hue="year", data=df, ax=ax2)
    score_vs_social_graph.set(xlabel="Happiness Score",
                              ylabel="Social Support",
                              title="Happiness Score vs Social Support")
    score_vs_life_graph = sns.scatterplot(x='life_expectancy',
                                          y='score',
                                          hue="year", data=df, ax=ax3)
    score_vs_life_graph.set(xlabel="Happiness Score",
                            ylabel="life_expectancy",
                            title="Happiness Score vs life expectancy")
    score_vs_freedom_graph = sns.scatterplot(x='freedom', y='score',
                                             hue="year", data=df, ax=ax4)
    score_vs_freedom_graph.set(xlabel="Happiness Score",
                               ylabel="Freedom to make Life Choices",
                               title="Happiness Score vs Freedom")
    plt.savefig('scorerelation.png')


def map_plot(world_data: gpd.GeoDataFrame) -> None:
    yeardata = world_data.groupby('country')[['score', 'GDP_log', 'social',
                                              'life_expectancy',
                                              'freedom']].mean().copy()
    filter_world = world_data[['SUBUNIT', 'geometry']]
    yeardata = yeardata.merge(filter_world, left_on='country',
                              right_on='SUBUNIT', how='left')
    yeardata_gpd = gpd.GeoDataFrame(yeardata)
    yeardata_gpd.from_features(yeardata_gpd.set_index("SUBUNIT"), crs='WGS84')

    fig = px.choropleth_mapbox(yeardata_gpd, geojson=yeardata_gpd.geometry,
                               locations=yeardata_gpd.index,
                               color='score',
                               color_continuous_scale="Viridis",
                               range_color=(2, 10),
                               center={'lat': 47.65749, 'lon': -122.30385},
                               mapbox_style="carto-positron",
                               zoom=3, opacity=0.3,
                               hover_name='SUBUNIT',
                               labels={'score': 'Average Score'},
                               title='Average Happiness Score'
                                     ' Map (2018 - 2021)'
                               )

    fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                      coloraxis_colorbar={'title':
                                          'Average Score (2018-2021)'})

    fig.show()


def split_data(data: gpd.GeoDataFrame) -> list:
    split = []
    # For Logistic Model
    X = data[['CONTINENT', 'GDP_log', 'social', 'life_expectancy', 'freedom']]
    features = pd.get_dummies(X, columns=["CONTINENT"], drop_first=True)
    data['above'] = np.where(data.score > (data['score'].mean()), 1, 0)
    lo_labels = data['above']
    # For Linear Model
    num_features = data[['GDP_log', 'social', 'life_expectancy', 'freedom']]
    li_labels = data['score']
    split = [(features, lo_labels), (num_features, li_labels)]
    return split


def marginal_effect(data: gpd.GeoDataFrame) -> None:
    print('Calculate marginal effect using logistic model')
    average = data['score'].mean()
    data['aboveaverage'] = np.where(data.score > average, 1, 0)
    m = smf.logit('aboveaverage ~ C(CONTINENT) + GDP_log +'
                  'social + life_expectancy + freedom', data=data).fit(
                  kwargs={"Warning": False}
                  )
    print(m.get_margeff().summary())
    print()


def logistic_model_generate(features_train, features_test,
                            labels_train, labels_test) -> None:
    # Training the model
    print("Training Logistic Model using Sklearn:")
    m = LogisticRegression(max_iter=2000)
    m.fit(features_train, labels_train)
    print("Finished training the model")
    print()
    # Predict using the model
    yhat = m.predict(features_test)
    # Scores
    confution_matrix = confusion_matrix(labels_test, yhat)
    accuracy = accuracy_score(labels_test, yhat)
    precision = precision_score(labels_test, yhat)
    recall = recall_score(labels_test, yhat)
    f_score = f1_score(labels_test, yhat)
    mse = mean_squared_error(labels_test, yhat)
    # Results
    print("The confusion matrix is:")
    print(confution_matrix)
    print()
    print("The accuracy is", accuracy)
    print("The precision is", precision)
    print("The recall is", recall)
    print("The F_score is", f_score)
    print("The mse is", mse)
    print()


def linear_model_generate(features_train, features_test,
                          labels_train, labels_test) -> LinearRegression:
    # Training the model
    print("Training linear Model using Sklean:")
    m = LinearRegression()
    m.fit(features_train, labels_train)
    print("Finished training the model")
    print()
    # Result
    intercept = np.round(m.intercept_, 5)
    coef = np.round(m.coef_, 5)
    print('Happiness score prediction Equation:')
    print(intercept, '+', coef[0], 'x GDP_log',
          '+', coef[1], 'x social', '+', coef[2], 'x life_expectancy',
          '+', coef[3], 'x freedom')
    # Predict using the model
    yhat = m.predict(features_test)
    # Scores
    mse = mean_squared_error(labels_test, yhat)
    # Results
    print("The mse is", mse)
    print()


# Main
def main():
    print('Main Method')

    # Load data
    df = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/'
                     'cse-163-group-project/main/data/data.csv')
    data = clean_data(df)
    print('complete clean')

    # Score distribution
    score_distr(data)
    # Score vs GDP Graph
    score_plot(data)

    # geodataframe
    WORLD_FILE = ('https://github.com/WanjiaRuan/cse-163-group-project/'
                  'blob/fd4af03bcaa683b7849afd75b7b6ee9acbfc0de9/data/'
                  'world.zip?raw=true')
    world = gpd.read_file(WORLD_FILE)[['NAME', 'SUBUNIT', 'SUBREGION',
                                       'CONTINENT', 'POP_EST', 'geometry']]
    # join / merge dataset with geodataframe
    world_data = join_data(data, world)
    print('complete join')

    # Map
    map_plot(world_data)
    print('complete plot')

    # Machine Learning Model
    print("Machine Learning")
    print('Split Data')
    linear_features, linear_labels = split_data(world_data)[1]
    li_features_train, li_features_test, li_labels_train, li_labels_test = \
        train_test_split(linear_features, linear_labels, test_size=0.25)
    logistic_features, logistic_labels = split_data(world_data)[0]
    lo_features_train, lo_features_test, lo_labels_train, lo_labels_test = \
        train_test_split(logistic_features, logistic_labels, test_size = 0.25)
    marginal_effect(world_data)
    logistic_model_generate(lo_features_train, lo_features_test,
                            lo_labels_train, lo_labels_test)
    linear_model_generate(li_features_train, li_features_test,
                          li_labels_train, li_labels_test)

if __name__ == '__main__':
    main()