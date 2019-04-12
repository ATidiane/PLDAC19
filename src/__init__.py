#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import date

import pandas as pd

from utils import remove_anomalies, sep_days, sep_month, sep_wde

"""_________________________________Constants_______________________________"""
"""_________________________________________________________________________"""

# data_repo = '/local/balde/datasets/stiflearning/'
# dep_to_keep = ['75', '77', '78', '91', '92', '93', '94', '95']

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

dict_w = remove_anomalies(dict_dates, to_del)
dict_wd_final = remove_anomalies(dict_wd, to_del)

dict_wd_octf = sep_month(dict_wd_final, 10)
dict_wd_novf = sep_month(dict_wd_final, 11)
dict_wd_decf = sep_month(dict_wd_final, 12)
