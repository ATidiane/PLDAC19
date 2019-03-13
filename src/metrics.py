#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

import numpy as np


def r2_score(y, y_pred):
    """ Returns the R² score.

    :param y: real values
    :param y_pred: predicted values
    :returns: a number
    :rtype: Float

    """

    return 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)


def mae(y, y_pred):
    """ Evaluate regression models by looking at how far there are off from the
    real Y, this metric is robust to outliers, thanks to the use of absolute
    value.

    :param y: real values
    :param y_pred: predicted values
    :returns: a number
    :rtype: Float

    """

    return sum(list(map(lambda p, q: abs(p - q), y, y_pred))) / len(y)


def mape(y, y_pred):
    """ Is the percentage equivalent of MAE, it describes the magnitude of the
    residuals as well as MAE and also has a clear interpretation since
    percentages are easier to conceptualize.

    :param y: real values
    :param y_pred: predicted values
    :returns: a number
    :rtype: Float

    """

    return 100 * np.mean(np.abs((y - y_pred) / y))


def mpe(y, y_pred):
    """ Exactly like the MAPE metrics but it lacks of absolute value operation.
    It's actually it's absence that makes MPE useful. It allows us to see if
    our model systematically underestimates(more negative error) or
    overestimates(more positive error).

    :param y: real values
    :param y_pred: predicted values
    :returns: a number
    :rtype: Float

    """

    return 100 * np.mean((y - y_pred) / y)


"""____________________________________Main_________________________________"""
"""_________________________________________________________________________"""


def main():
    pass


if __name__ == '__main__':
    main()
