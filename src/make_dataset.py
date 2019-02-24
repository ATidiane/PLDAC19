#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

import sys
from datetime import time

import numpy as np
import pandas as pd

from connection import Connection
from utils import *
from utils import save_in_pkl

sys.path.insert(0, '../src/')


def discretise_validations(stations, dates, times):

    def count(cursor, station, date, start, end):
        query = ("SELECT count(*) FROM validation "
                 "WHERE station_id = {} AND operation_date = \"{}\" "
                 "AND time between \"{}\" AND \"{}\"".format(
                     station, date, start, end))

        cursor.execute(query)
        return cursor.fetchall()[0][0]

    def count_last(cursor, station, date, start):
        query = ("SELECT count(*) FROM validation "
                 "WHERE station_id = {} AND operation_date = \"{}\" "
                 "AND time >= \"{}\"".format(station, date, start))

        cursor.execute(query)
        return cursor.fetchall()[0][0]

    connection = Connection()
    cursor = connection.cursor()

    print("Creating Dataset...")
    matrix = []
    for i, date in enumerate(dates):
        print("    ", i, date)
        matrix.append([])
        for j, station_id in enumerate(stations):
            print("        ", j, station_id)
            matrix[i].append([])
            for k, temps in enumerate(times):
                if k == len(times) - 1:
                    matrix[i][j].append(
                        count_last(
                            cursor,
                            station_id,
                            date,
                            temps))
                else:
                    matrix[i][j].append(
                        count(cursor, station_id, date, temps, times[k + 1]))

        print("Saving {}.npy".format(date))
        np.save("../datasets/{}".format(date), np.array(matrix[i]))

    np.save("../datasets/all_matrix", np.array(matrix))
    print("Done...")


if __name__ == "__main__":
    stations_mode = load_pkl("../datasets/stations_mode.pkl")

    metro_stations = [k for k, v in stations_mode.items() if v == 3]

    dates = pd.date_range(start="2015-10-01", end="2015-12-31").date

    times = pd.date_range(
        start="2015-10-01",
        end="2015-10-01 23:59:59",
        freq="15min").time

    discretise_validations(metro_stations, dates, times)
