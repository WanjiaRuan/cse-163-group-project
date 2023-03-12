"""
CSE 163 Project
Author: Xingyuan Zhao, Mariana Li Chen, Wanjia Ruan
Implement function for testing codes in "deliverable.py"
There are three assert_equal testing functions which
testing different aspects of the main code.
"""
# import files
import pandas as pd
import geopandas as gpd
from cse163_utils import assert_equals
import deliverable as d
from sklearn.model_selection import train_test_split


def test_clean_data(one: pd.DataFrame, two: pd.DataFrame) -> None:
    """This function test the shape of then clean_data function"""
    assert_equals((137, 7), one.shape)
    assert_equals((249, 7), two.shape)


def test_join_data(one_gpd: gpd.GeoDataFrame,
                   two_gpd: gpd.GeoDataFrame) -> None:
    """ This function test the shape of the join_data function"""
    assert_equals((129, 13), one_gpd.shape)
    assert_equals((236, 13), two_gpd.shape)


def test_split_data(one_gpd: gpd.GeoDataFrame,
                    two_gpd: gpd.GeoDataFrame) -> None:
    """
    This function test the split_data function to check:
    1. If the function correctly split two GeoDataFrame into training
    and testing sets for logistic and linear regression model.
    2. Check the features and labels variable and their numbers to see
    if it splits correctly
    """
    # For GeoDataFrame One
    one_logistic_features, one_logistic_labels = d.split_data(one_gpd)[0]
    one_linear_features, one_linear_labels = d.split_data(one_gpd)[1]
    assert_equals(9, len(one_logistic_features.columns))
    assert_equals(one_gpd['above'], one_logistic_labels)
    assert_equals(4, len(one_linear_features.columns))
    assert_equals(one_gpd['score'], one_linear_labels)

    one_li_features_train, one_li_features_test, \
        one_li_labels_train, one_li_labels_test = \
        train_test_split(one_linear_features, one_linear_labels,
                         test_size=0.25)
    assert_equals(96, len(one_li_features_train))
    assert_equals(96, len(one_li_labels_train))
    assert_equals(33, len(one_li_features_test))
    assert_equals(33, len(one_li_labels_test))

    # For GeoDataFrame Two
    two_logistic_features, two_logistic_labels = d.split_data(two_gpd)[0]
    two_linear_features, two_linear_labels = d.split_data(two_gpd)[1]
    assert_equals(9, len(two_logistic_features.columns))
    assert_equals(two_gpd['above'], two_logistic_labels)
    assert_equals(4, len(two_linear_features.columns))
    assert_equals(two_gpd['score'], two_linear_labels)

    two_li_features_train, two_li_features_test, \
        two_li_labels_train, two_li_labels_test = \
        train_test_split(two_linear_features, two_linear_labels,
                         test_size=0.25)
    assert_equals(177, len(two_li_features_train))
    assert_equals(177, len(two_li_labels_train))
    assert_equals(59, len(two_li_features_test))
    assert_equals(59, len(two_li_labels_test))


def main():
    # datasets
    one = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/'
                      'cse-163-group-project/main/data/one.csv')
    two = pd.read_csv('https://raw.githubusercontent.com/WanjiaRuan/'
                      'cse-163-group-project/main/data/two.csv')
    WORLD_FILE = ('https://github.com/WanjiaRuan/cse-163-group-project/'
                  'blob/fd4af03bcaa683b7849afd75b7b6ee9acbfc0de9/data/'
                  'world.zip?raw=true')
    world = gpd.read_file(WORLD_FILE)[['NAME', 'SUBUNIT', 'SUBREGION',
                                       'CONTINENT', 'POP_EST', 'geometry']]
    # testing
    one = d.clean_data(one)
    two = d.clean_data(two)
    test_clean_data(one, two)
    one_gpd = d.join_data(one, world)
    two_gpd = d.join_data(two, world)
    test_join_data(one_gpd, two_gpd)
    test_split_data(one_gpd, two_gpd)
    print('Testing all passed.')


if __name__ == '__main__':
    main()
