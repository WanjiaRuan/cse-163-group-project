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


# Main
def main():
    print('Main Method')
    # data
    year2019 = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/cse-163-group-project/main/data/2019.csv')
    year2020 = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/cse-163-group-project/main/data/2020.csv')
    year2021 = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/cse-163-group-project/main/data/2021.csv')
    year2022 = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/cse-163-group-project/main/data/2022.csv')
    # join data
    data = [year2019, year2020, year2021, year2022]
    data = pd.concat(data)


if __name__ == '__main__':
    main()