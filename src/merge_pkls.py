#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

import glob
from collections import defaultdict

import numpy as np
import pandas as pd

from utils import load_pkl


def merge_pkls(stations, dates, path):
    """FIXME! briefly describe function

    :param stations:
    :param dates:
    :param path:
    :returns:
    :rtype:

    """

    list_files = sorted(glob.glob(path))

    matrix = np.zeros((len(dates), len(stations), len(list_files)))

    print("Starting merge...")
    for i, fichier in enumerate(list_files):
        print("    {}...".format(fichier))
        file_dic = defaultdict(int, load_pkl(fichier))
        for d, date in enumerate(dates):
            for s, station in enumerate(stations):
                matrix[d][s][i] = file_dic[(str(date), station)]

    print("Done...")
    return matrix


def main():
    # Variables declarations
    stations_mode = load_pkl("../datasets/stations_mode.pkl")

    metro_stations = [k for k, v in stations_mode.items() if v == 3]

    dates = pd.date_range(start="2015-10-01", end="2015-12-31").date

    # Merging all the pkl files in 6h_matrix_pkls into 6h_matrix.npy
    matrix_6h = merge_pkls(metro_stations, dates,
                           "../datasets/6h_matrix_pkls/*.pkl")
    np.save("../datasets/6h_matrix", matrix_6h)

    # Merging all the pkl files in 2h_matrix_pkls into 2h_matrix.npy
    matrix_2h = merge_pkls(metro_stations, dates,
                           "../datasets/2h_matrix_pkls/*.pkl")
    np.save("../datasets/2h_matrix", matrix_2h)

    # Merging all the pkl files in 15m_matrix_pkls into 15m_matrix.npy
    matrix_15m = merge_pkls(metro_stations, dates,
                            "../datasets/15m_matrix_pkls/*.pkl")
    np.save("../datasets/15m_matrix", matrix_15m)


if __name__ == "__main__":
    main()
