#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This file is used for preprocessing the data, removing unuseful columns,
taking care of categorical features by applying Label encoding or OneHot
encoding, normalizing if needed and so on.

Author: Ahmed Tidiane BALDE
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
from pyproj import Proj, transform

from constant import data_matrix_15m_complete, dict_w, subway_stations
from utils import generate_times, sep_days

"""_____________________________Generic Functions___________________________"""
"""_________________________________________________________________________"""


def delete_columns(data, columns_to_drop):
    """ Delete specified columns in columns_to_drop from the dataframe

    :param data: dataframe from which to delete columns
    :param columns_to_drop: list of columns to delete
    :returns: a dataframe
    :rtype: pandas.DataFrame

    """

    for column in columns_to_drop:
        if column in data.columns:
            data.drop(column, axis=1, inplace=True)

    return data


def exist_categorical(data):
    """ Return True if there's any categorical variable left, else False

    :param data: dataframe
    :returns: True of False
    :rtype: Boolean

    """

    return any(map(lambda col: isinstance(data.loc[0, col], str),
                   data.columns))


def categorical_columns(data):
    """ Return the list of categorical columns in the dataset if exists

    :param data: dataframe
    :returns: List of categorical columns
    :rtype: List

    """

    return list(filter(lambda col: isinstance(data[col][0], str),
                       data.columns))


def apply_to_columns(data, columns, function):
    """ Apply the given function to the given columns

    :param data: dataframe
    :param columns: columns on which to apply the function
    :param function: the function to apply
    :returns: dataframe
    :rtype: pandas.DataFrame

    """

    for col in columns:
        data[col] = data[col].map(function, na_action='ignore')

    return data


def keep_rows(data, column, values):
    """ Return a new dataframe that includes all rows where the value of a
    cell in the given column is in the list of given values

    :param data: dataframe
    :param column: name of the column
    :param values: list of unique values of the column to keep
    :returns: dataframe
    :rtype: pandas.DataFrame

    """

    data = data[data[column].isin(values)]
    data.index = range(data.shape[0])

    return data


def one_hot_encoder(data, columns, **params):
    """ Apply OneHot Encoding on the given columns of the dataframe

    :param data: dataframe
    :param columns: list of columns to encode
    :returns: dataframe
    :rtype: pandas.DataFrame

    """

    return pd.get_dummies(data, columns, **params)


def day_of_week(date):
    """ Convert the date format "2015-01-01" into a simple format, an integer.
    For instance: "2015-01-01" will return 4 because it was a Thursday.

    :param date: date in string format "2015-01-01"
    :returns: the corresponding day of week in number
    :rtype: integer

    """

    return datetime.strptime(
        date, "%Y-%m-%d").isoweekday()


def aggregate_time_hour(times):
    """ Convert the given time "16:15:01" into an integer, depending on the
    aggregration we sticked to.

    :param times: time in string format "16:15:01"
    :returns: the corresponding integer from our aggregation rule
    :rtype: integer

    """

    hour = time.strptime(times, "%H:%M:%S")[3]

    if hour in range(7, 10):
        return 1
    elif hour in range(10, 17):
        return 2
    elif hour in range(17, 21):
        return 3
    else:
        return 4


def convert_lambert93_to_latlong(x, y):
    """ Convert station coordinates which are in Lambert-93 to latitude and
    longitude, in order to have the same units of the polygons.

    :param x: Longitude in Lambert-93
    :param y: Latitude in Lambert-93
    :returns: Longitude and latitude in degrees
    :rtype: tuple(double, double)

    """

    inProj = Proj(init='epsg:2154')
    outProj = Proj(init='epsg:4326')

    return transform(inProj, outProj, x, y)


def repl_neg_pred(y):
    """ Replace negative values of regression models predictions by 0

    :param y: predicted values
    :returns: transformed predicted values
    :rtype: np.array

    """

    return np.array(list(map(lambda v: 0 if v < 0 else v, y)))


def normalize(X):
    """ Returns a normalized (between 0 and 1) array of X

    :param X: independent variables, features.
    :returns: a normalized X
    :rtype: np.array

    """

    return (X - X.min()) / (X.max() - X.min())


def normalize_by_column(X):
    """ Returns a normalized (between 0 and 1) array of X. Normalizes each
    column separatly.

    :param X: independent variables, features.
    :returns: a normalized X
    :rtype: np.array

    """

    for i in range(X.shape[1]):
        tmp = X[:, i]
        maxi, mini = tmp.max(), tmp.min()
        X[:, i] = (tmp - mini) / (maxi - mini)

    return X


def denormalize_y(y, maxi, mini):
    """ Denormalize dependent variable.

    :param y: dependent variable
    :param maxi: maximum of original variable before normalization
    :param mini: minimum of original variable before normalization
    :returns: a denormalized variable
    :rtype: np.array

    """

    return y * (maxi - mini) + mini


def missing_ratio(data):
    """ Returns a new dataframe with all features as index and their
    corresponding ratio of missing values

    :param data: dataframe
    :returns: dataframe of missing values ratio for each feature
    :rtype: pandas.DataFrame

    """

    data_na = (data.isnull().sum() / len(data)) * 100
    all_data_na = data_na.drop(
        data_na[data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})

    return missing_data


def normalize_to_panel(X, X_min_max, station_id, a=0, b=1):
    """

    :param X:
    :param X_min_max:
    :param station_id:
    :param a:
    :param b:

    """
    mini, maxi = X_min_max.loc[station_id]
    return (a + ((X.loc[:, station_id] - mini)
                 * (b - a)) / (maxi - mini)).values


def denormalize_to_panel(X, X_min_max, station_id, a=0, b=1):
    """

    :param X:
    :param X_min_max:
    :param station_id:
    :param a:
    :param b:

    """
    mini, maxi = X_min_max.loc[station_id]
    return (((X.loc[:, station_id] - a) * (maxi - mini)) /
            (b - a) + mini).values


def normalize_panel_data(X_train, X_test, del_hours=4, a=-1, b=1):
    """FIXME! briefly describe function

    :param X_train:
    :param X_test:
    :param a:
    :param b:
    :returns:
    :rtype:

    """
    # Normalization between a and b
    X_min_max = X_train.apply(
        lambda x: (
            x.min().min(), x.max().max()), axis=(
            0, 2))

    Xn_train = X_train.apply(lambda x: a + ((x - x.min().min()) * (b - a)) / (
        x.max().max() - x.min().min()), axis=(0, 2)).transpose(2, 0, 1)

    # Normalize X_test
    Xn_test = pd.Panel(np.array(list(map(lambda station_id: normalize_to_panel(X_test,
                                                                               X_min_max,
                                                                               station_id, a=a, b=b),
                                         X_test.transpose(1, 0, 2)))).transpose(2, 0, 1),
                       items=list(X_test.items),
                       major_axis=subway_stations,
                       minor_axis=generate_times("15min")[(del_hours * 4):])

    return Xn_train, Xn_test, X_min_max


def del_split_norm(
        dico,
        test_size,
        diff_days,
        del_hours=4,
        stations_to_del=[
            4113,
            20868],
        a=-1,
        b=1):
    """FIXME! briefly describe function

    :param dico:
    :param size:
    :param diff_days:
    :param del_hours:
    :param stations_to_del:
    :param 20868]:
    :param a:
    :param b:
    :returns:
    :rtype:

    """

    data_matrix_15m = data_matrix_15m_complete.iloc[:, :, del_hours * 4:]
    X = data_matrix_15m.loc[dico.values()]
    X.drop(columns=stations_to_del, inplace=True)

    try:
        for s in stations_to_del:
            subway_stations.remove(s)
    except BaseException:
        pass

    # Split
    ddico = sep_days(dico, diff_days)
    for i in range(len(ddico)):
        ddico[i] = list(ddico[i].values())

    i_test = []

    if test_size <= diff_days:
        for i in range(test_size):
            chosen_ind = np.random.choice(ddico[i])
            i_test.append(chosen_ind)
            print(ddico[i], "\n")
            ddico[i].remove(chosen_ind)
            print(ddico[i])
    else:
        r = test_size % diff_days
        d = test_size // diff_days
        for i in range(diff_days):
            chosen_inds = np.random.choice(ddico[i], d, replace=False)
            i_test.extend(chosen_inds)
            for e in chosen_inds:
                ddico[i].remove(e)

        for i in np.random.choice(diff_days, r, replace=False):
            chosen_ind = np.random.choice(ddico[i])
            i_test.append(chosen_ind)
            ddico[i].remove(chosen_ind)

    # Flatten the dictionnary
    i_train = [j for i in ddico for j in ddico[i]]

    X_train = X[sorted(i_train)]
    X_test = X[sorted(i_test)]

    # Normalization
    Xn_train, Xn_test, X_min_max = normalize_panel_data(
        X_train, X_test, del_hours=del_hours, a=a, b=b)

    return X_train, X_test, Xn_train, Xn_test, X_min_max


""" ____________________________________Main_________________________________"""
"""_________________________________________________________________________"""


def main():
    X_train, X_test, Xn_train, Xn_test, X_min_max = del_split_norm(
        dict_w, test_size=14, diff_days=7)

    print(X_train.items, "\n")
    print(Xn_train.items, "\n")

    print(X_test.items, "\n")
    print(Xn_test.items)


if __name__ == "__main__":
    main()
