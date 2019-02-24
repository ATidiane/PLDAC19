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


def discretise_validations_by_time(stations, times):

    def count(cursor, stations, start, end):
        query = ("SELECT operation_date, station_id, count(*) FROM validation "
                 "WHERE station_id in {} "
                 "AND time between \"{}\" AND \"{}\" "
                 "GROUP BY operation_date, station_id".format(
                     stations, start, end))

        cursor.execute(query)
        return {(operation_date, station_id): nb_validations for
                operation_date, station_id, nb_validations in
                cursor.fetchall()}

    def count_last(cursor, station, start):
        query = ("SELECT operation_date, station_id, count(*) FROM validation "
                 "WHERE station_id in {} "
                 "AND time >= \"{}\" "
                 "GROUP BY operation_date, station_id".format(
                     stations, start))

        cursor.execute(query)
        return {(operation_date, station_id): nb_validations for
                operation_date, station_id, nb_validations in
                cursor.fetchall()}

    connection = Connection()
    cursor = connection.cursor()

    print("Creating Dataset...")

    for i, temps in enumerate(times):
        print("    ", i, temps)
        if i == len(times) - 1:
            save_in_pkl(count_last(cursor, stations, temps),
                        "../datasets/2h_{}.pkl".format(temps))
        else:
            save_in_pkl(
                count(cursor,
                      stations,
                      temps,
                      times[i + 1]),
                "../datasets/2h_{}.pkl".format(temps))

    print("Done...")


if __name__ == "__main__":
    stations_mode = load_pkl("../datasets/stations_mode.pkl")

    metro_stations = [k for k, v in stations_mode.items() if v == 3]

    dates = pd.date_range(start="2015-10-01", end="2015-12-31").date

    times = pd.date_range(
        start="2015-10-01",
        end="2015-10-01 23:59:59",
        freq="2h").time

    discretise_validations_by_time(tuple(metro_stations), times)
