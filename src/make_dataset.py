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

    connection = Connection()
    cursor = connection.cursor()

    def count(cursor, station, date, start, end):
        query = ("SELECT count(*) FROM validations "
                 "WHERE station_id = {} AND operation_date = {} "
                 "AND time between {} AND {}".format(
                     station, date, start, end))

        cursor.excecute(query)
        return cursor.fetchall()

    def count_last(cursor, station, date, start):
        query = ("SELECT count(*) FROM validations "
                 "WHERE station_id = {} AND operation_date = {} "
                 "AND time >= {}".format(station, date, start))

        cursor.excecute(query)
        return cursor.fetchall()

    matrix = []
    for i, date in enumerate(dates):
        matrix.append([])
        for j, station in enumerate(metro_stations):
            matrix[i].append([])
            for k, temps in enumerate(times):
                if k == len(times):
                    matrix[i][j].append(
                        count_last(
                            cursor,
                            station,
                            date,
                            temps))
                else:
                    matrix[i][j].append(
                        count(cursor, station, date, temps, times[k + 1]))


if __name__ == "__main__":
    stations_mode = load_pkl("../datasets/stations_mode.pkl")

    metro_stations = [k for k, v in stations_mode.items() if v == 3]

    dates = pd.date_range(start="2015-10-01", end="2015-10-01").date

    for d in dates:
        print(d)

    times = pd.date_range(
        start="2015-10-01",
        end="2015-10-01 23:59:59",
        freq="15min").time
