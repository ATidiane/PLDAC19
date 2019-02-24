#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This file contains all the useful functions, used in the process of
completing this project or not. Everything useful for making an optimized code.

Author: Ahmed Tidiane BALDE
"""

import io
import pickle

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


"""____________________________________Main_________________________________"""
"""_________________________________________________________________________"""


def main():
    pass


if __name__ == '__main__':
    main()
