""" 
CSE 163 Project
Author: Xingyuan Zhao, Mariana Li Chen, Wanjia Ruan
implement function for the final group project, this is the file
that contains all codes that produce expected output based on the
World Happiness Report dataset and the research questions which tends
to provide a deeper analysis for the Happiness Score and its indexes.
"""
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
    """
    This function takes the parameter df and return a dataset
    with the selected 'Ladder Score' and four indexes from 2018 to  
    2019 with dropped NA values and renamed columns.
    """
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
    """
    This function takes two parameters, Dataset and GeoDataFrame,
    and join the two parameters to return a New GeoDataFrame for
    each country's Happiness Score from 2018 to 2021 wwith dropped NA
    values.
    """

    # Combine world and happiness datasets
    world_data = data.merge(world, left_on='country',
                            right_on='SUBUNIT', how='left')
    world_data = world_data.dropna()
    world_data_gpd = gpd.GeoDataFrame(world_data)
    return world_data_gpd


def score_distr(df: pd.DataFrame) -> None:
    """
    This function takes one parameter df and return four different
    histograms that shows the happiness score distribution between
    the years 2018 to 2021 saved in the same png file.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()
    years = [2018, 2019, 2020, 2021]
    for i, year in enumerate(years):
        year_df = df[df['year'] == year]
        plot = sns.histplot(data=year_df, x="score",
                            color="purple", kde=True, ax=axs[i])
        plot.set(xlabel="Happiness Score", title=f"(Happiness Score {year})")
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.3, hspace=0.3)
    fig.savefig("score_distribuotion.png")


def score_plot(df: pd.DataFrame) -> None:
    """
    This function takes one parameter df and return four different
    scatterplots that shows the correlation between happiness score
    and each of the social indexes between years 2018 to 2021 saved in
    the same png file.
    """
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
    """
    This function take one parameter world_data (GeoDataFrame) and return
    an interactive map colored gradiently to indicate the distribution of
    the scores.
    """
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
    """
    This function split the data into training set and test set, and create
    logisitic regression model for the future prediction.
    """
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
    """
    This is the function that creates the mariginal effects of the
    logisitic regression model and show the summary.
    """
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
    """
    This function creats the logisitic regression model using sklearn
    library abd train the model. Getting the confusion matrix and the
    accuracy, precision, recall, f-score, and MSE.
    """
    # Training the model
    m = LogisticRegression(max_iter=2000)
    m.fit(features_train, labels_train)
    print()

    # Predict using training data
    yhat_train = m.predict(features_train)

    # Predict using testing data
    yhat_test = m.predict(features_test)

    # Scores
    confution_matrix = confusion_matrix(labels_test, yhat_test)
    test_acc = accuracy_score(labels_test, yhat_test)
    train_acc = accuracy_score(labels_train, yhat_train)
    precision = precision_score(labels_test, yhat_test)
    recall = recall_score(labels_test, yhat_test)
    f_score = f1_score(labels_test, yhat_test)
    mse = mean_squared_error(labels_test, yhat_test)

    # Results
    print("The testing result confusion matrix:")
    print(confution_matrix)
    print()
    print("Testing Accuracy:", test_acc)
    print('Training Accuracy:', train_acc)
    print("The Testing Precision:", precision)
    print("The TestingRecall:", recall)
    print("The TestingF_score:", f_score)
    print("The Testing MSE:", mse)
    print()


def linear_model_generate(features_train, features_test,
                          labels_train, labels_test) -> None:
    
    """
    This is the function using skklean create the linear regression
    model and using test set to get the MSE and the intercept, and
    the coeeficients for the variables.

    """
    # Training the model
    m = LinearRegression()
    m.fit(features_train, labels_train)
    print()

    # Result
    intercept = np.round(m.intercept_, 5)
    coef = np.round(m.coef_, 5)
    print('Happiness score prediction Equation:')
    print(intercept, '+', coef[0], 'x GDP_log',
          '+', coef[1], 'x social', '+', coef[2], 'x life_expectancy',
          '+', coef[3], 'x freedom')
    print()

    # Predict using training data
    yhat_train = m.predict(features_train)

    # Predict using the testing data
    yhat_test = m.predict(features_test)

    # Scores
    test_mse = mean_squared_error(labels_test, yhat_test)
    train_mse = mean_squared_error(labels_train, yhat_train)

    # Results
    print('Training MSE:', train_mse)
    print("Testing MSE:", test_mse)
    print()


# Main
def main():
    print('--------------  Welcome to our CSE 163 Project: '
          'World Happiness Report  --------------')
    print()

    # Load data
    print('Loading and cleaning happiness data...')
    df = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/'
                     'cse-163-group-project/main/data/data.csv')
    data = clean_data(df)
    print('Complete Cleaning!')
    print()

    # Score distribution
    print('----------------- Plotting Histograms -----------------')
    score_distr(data)
    # Score vs GDP Graph
    print('--------------- Plotting Scatter Plots ---------------')
    print()
    score_plot(data)

    # geodataframe
    print('Joining World GeoDataFrame with existing DataFrame...')
    WORLD_FILE = ('https://github.com/WanjiaRuan/cse-163-group-project/'
                  'blob/fd4af03bcaa683b7849afd75b7b6ee9acbfc0de9/data/'
                  'world.zip?raw=true')
    world = gpd.read_file(WORLD_FILE)[['NAME', 'SUBUNIT', 'SUBREGION',
                                       'CONTINENT', 'POP_EST', 'geometry']]
    # join / merge dataset with geodataframe
    world_data = join_data(data, world)
    print('Complete joining!')
    print()

    # Map
    print('--------------- Showing an interactive world map ---------------')
    map_plot(world_data)
    print('Complete Plotting!')
    print()

    # Machine Learning Model
    print("----------------------- Machine Learning -----------------------")
    print('Splitting Data...')
    print()

    linear_features, linear_labels = split_data(world_data)[1]
    li_features_train, li_features_test, li_labels_train, li_labels_test = \
        train_test_split(linear_features, linear_labels, test_size=0.25)

    logistic_features, logistic_labels = split_data(world_data)[0]
    lo_features_train, lo_features_test, lo_labels_train, lo_labels_test = \
        train_test_split(logistic_features, logistic_labels, test_size=0.25)

    print('Calculating marginal effect using logistic model...')
    print('---------------------- Results shows below ----------------------')
    marginal_effect(world_data)

    print('------------------------ Logistic Model ------------------------')
    logistic_model_generate(lo_features_train, lo_features_test,
                            lo_labels_train, lo_labels_test)

    print('------------------------ Linear Model ------------------------')
    linear_model_generate(li_features_train, li_features_test,
                          li_labels_train, li_labels_test)
    print('Finish.')


if __name__ == '__main__':
    main()
