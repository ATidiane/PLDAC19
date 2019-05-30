#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""


from __init__ import *
from utils import (generate_times, load_pkl, remove_anomalies, sep_days,
                   sep_month, sep_wde)

"""_________________________________Constants_______________________________"""
"""_________________________________________________________________________"""


dates = pd.date_range(start="2015-10-01", end="2015-12-31").date

dict_dates = {i: d for i, d in zip(range(len(dates)), dates)}
ddict_days = sep_days(dict_dates, diff_days=7)

dict_wd = sep_wde(dict_dates, week="days")
dict_we = sep_wde(dict_dates, week="end")

dict_wd_oct = sep_month(dict_wd, 10)
dict_wd_nov = sep_month(dict_wd, 11)
dict_wd_dec = sep_month(dict_wd, 12)

nov_del = [date(2015, 11, 11), date(2015, 11, 29), date(2015, 11, 30)]
vacs = pd.date_range(start="2015-12-21", end="2015-12-31").date.tolist()
to_del = nov_del + vacs

# Dates without anomalies
dict_w = remove_anomalies(dict_dates, to_del)
dict_wd_final = remove_anomalies(dict_wd, to_del)

dict_wd_octf = sep_month(dict_wd_final, 10)
dict_wd_novf = sep_month(dict_wd_final, 11)
dict_wd_decf = sep_month(dict_wd_final, 12)

# cluster of days after anomalies removed
ddict_w = sep_days(dict_w, diff_days=7)
ddict_wd = sep_days(dict_wd, diff_days=5)

# Global variables
global data_matrix_15m_complete, subway_stations

# Different subway_stations
stations_mode = load_pkl("../datasets/stations_mode.pkl")
subway_stations = [k for k, v in stations_mode.items() if v == 3]

# Load number of validations data
matrix_15m = np.load("../datasets/15m_matrix.npy")

data_matrix_15m_complete = pd.Panel(matrix_15m,
                                    items=dates,
                                    major_axis=subway_stations,
                                    minor_axis=generate_times("15min")
                                    )
