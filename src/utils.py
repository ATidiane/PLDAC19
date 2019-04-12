#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This file contains all the useful functions, used in the process of
completing this project or not. Everything useful for making an optimized code.

Author: Ahmed Tidiane BALDE
"""

import io
import pickle

import numpy as np
import pandas as pd


def load_pkl(filename):
    """ Read pkl file

    :param filename: file name
    :returns: a dictionary
    :rtype: dict

    """

    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_in_pkl(obj, filename):
    """ Save pkl file

    :param obj: object to save, usually a dictionary
    :param filename: file to save the object in
    :returns: None
    :rtype: Void

    """

    with open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)


def read_file_txt(filename):
    """ Read text file !

    :param filename: file name
    :returns: list of sentences
    :rtype: list

    """

    f = io.open(filename, 'rU', encoding='utf-8')
    sentences = [line for line in f.readlines()]

    return sentences


def read_file_pd_ext(filename, **params):
    """ Read supported extensions with pandas, if an extension is
    not supported then return an exception.

    :param filename: filename
    :returns: a dataframe containing the file
    :rtype: pandas.DataFrame

    """

    pd_ext = {
        'csv': pd.read_csv,
        'json': pd.read_json,
        'html': pd.read_html,
        'xls': pd.read_excel,
        'h5': pd.read_hdf,
        'feather': pd.read_feather,
        'msg': pd.read_msgpack,
        'dta': pd.read_stata,
        'pkl': pd.read_pickle}

    file_ext = filename.split('.')[-1]

    try:
        return pd_ext[file_ext](filename, **params)
    except BaseException:
        print("Extension not supported by pandas")


def generate_times(freq, debut="2015-10-01", fin="2015-10-01"):
    """FIXME! briefly describe function

    :param frep:
    :returns:
    :rtype:

    """
    times = pd.date_range(
        start=debut,
        end=fin + " 23:59:59",
        freq=freq).time

    return times


def sep_days(dico, diff_days=7):
    """

    :param dico:
    :param diff_days:
    :return:
    :rtype:

    """

    return {d: dict(filter(lambda dd: dd[1].weekday() == d,
                           zip(dico, dico.values())))
            for d in range(diff_days)}


def sep_wde(dico, week="days"):
    """

    :param dico:
    :param week:
    :return:
    :rtype:

    """

    if week.lower() == "days":
        return dict(filter(lambda dd: dd[1].weekday() < 5,
                           zip(dico, dico.values())))
    elif week.lower() == "end":
        return dict(filter(lambda dd: dd[1].weekday() >= 5,
                           zip(dico, dico.values())))
    else:
        raise ValueError("Wrong value for parameter \"week\"\n \
        Only takes two differents values : \"days\" or \"end\"")


def sep_month(dico, month_num=10):
    """

    :param dico:
    :param month_num:
    :return:
    :rtype:

    """

    return dict(filter(lambda dd: dd[1].month == month_num,
                       zip(dico, dico.values())))


def remove_anomalies(dico, anomalies):
    """

    :param dico:
    :param anomalies:
    :return:
    :rtype:

    """

    return dict(filter(lambda dd: dd[1] not in anomalies,
                       zip(dico, dico.values())))


def create_panel_pred(ar_preds, X_test, t, order,
                      subway_stations, del_hours=0):
    """

    """

    wd_testorder_15m = X_test.iloc[:, :, order:]

    return pd.Panel(np.array(ar_preds[t - 1]),
                    items=wd_testorder_15m.items,
                    major_axis=subway_stations,
                    minor_axis=generate_times("15min")[(del_hours * 4) + order:])


"""____________________________________Main_________________________________"""
"""_________________________________________________________________________"""


def main():
    pass


if __name__ == '__main__':
    main()
