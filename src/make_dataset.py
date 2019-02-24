import sys

import numpy as np
import pandas as pd

from utils import *

sys.path.insert(0, '../src/')


stations_mode = load_pkl("../datasets/stations_mode.pkl")

metro_stations = [k for k, v in stations_mode.items() if v == 3]

dates = pd.date_range(start="2015-10-01", end="2015-12-31").date

times = pd.date_range(start="2015-10-01", end="2015-10-02", freq="15min").time

for t in times:
    print(t - 1)
