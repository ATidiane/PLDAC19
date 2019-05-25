#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

import numpy as np

from constant import ddict_days


class Baseline:
    def __init__(self, level=None, first_ndays=7):
        """ Initialization of Baseline parameters

        :param level: level of computing, means : if level is None the rmse is
        computed on the whole Train Set, if level is equal to "S", then the
        metric is applied separatly for each station, if it is equal to "J",
        it is applied separetly for each day : Lundi, Mardi, etc..., if it is
        equal to "SJ", then for each station and each day, we aggregate the
        values and compute the metric. And if level is none of the listed
        values above then, we consider that it is None, by default.

        :param first_ndays:
        :return:
        :rtype:

        """

        self.first_ndays = first_ndays
        self.level = level
        if self.level not in [
            None,
            "s",
            "S",
            "j",
            "J",
            "sj",
            "SJ",
            "Sj",
                "sJ"]:
            self.level = None

    def fit(self, datax):
        """FIXME! briefly describe function

        :param datax:
        :returns:
        :rtype:

        """

        if self.level is None:
            self.mean = datax.mean().mean(axis=1)
        elif self.level.lower() == "s":
            self.mean = datax.mean(axis=0)
        elif self.level.lower() == "j":
            self.mean = []
            for d in range(self.first_ndays):
                exist_ind = list(
                    set(ddict_days[d].values()) & set(datax.items))
                self.mean.append(datax[exist_ind].mean().mean(axis=1))
        elif self.level.lower() == "sj":
            self.mean = []
            for d in range(self.first_ndays):
                exist_ind = list(
                    set(ddict_days[d].values()) & set(datax.items))
                self.mean.append(datax[exist_ind].mean(axis=0))
        else:
            raise ValueError("Unknown value for level attribute, \
            try: s, j, sj or None")

    def predict(self, datax):
        """FIXME! briefly describe function

        :param datax:
        :returns:
        :rtype:

        """

        if self.level is None:
            return datax.apply(lambda x: self.mean, axis="minor")

        elif self.level.lower() == "s":
            return datax.apply(lambda x: self.mean, axis=(1, 2))

        elif self.level.lower() == "j":
            datax_copy = datax.copy()
            for d in range(self.first_ndays):
                exist_ind = list(
                    set(ddict_days[d].values()) & set(datax.items))
                # print(exist_ind, "\n")
                datax_copy.update(datax_copy[exist_ind].apply(
                    lambda x: self.mean[d], axis="minor"))

            return datax_copy

        elif self.level.lower() == "sj":
            datax_copy = datax.copy()
            for d in range(self.first_ndays):
                exist_ind = list(
                    set(ddict_days[d].values()) & set(datax.items))
                datax_copy.update(datax_copy[exist_ind].apply(
                    lambda x: self.mean[d], axis=(1, 2)))

            return datax_copy

        else:
            raise ValueError("Unknown value for level attribute, \
            try: s, j, sj or None")

    def metrics_score(self, df_X, X_pred):
        """FIXME! briefly describe function

        :param df_X:
        :param X_pred:
        :returns:
        :rtype:

        """

        r2_score = 1 - (np.sum((df_X.values - X_pred)**2) /
                        np.sum((df_X.values - df_X.values.mean())**2))

        mae_mat = np.abs(df_X.values - X_pred) / df_X.values
        mae_mat = np.where(mae_mat == np.inf, 0, mae_mat)
        mae = np.nanmean(mae_mat)

        mape = 100 * mae

        mpe_mat = (df_X.values - X_pred) / df_X.values
        mpe_mat = np.where(mpe_mat == np.inf, 0, mpe_mat)
        mpe_mat = np.where(mpe_mat == -np.inf, 0, mpe_mat)
        mpe = 100 * np.nanmean(mpe_mat)

        rmse = np.sqrt(((X_pred - df_X.values)**2).mean())

        mse = ((df_X.values - X_pred)**2).mean()

        return [r2_score, rmse, mse, mae, mape, mpe]

    def score(self, datax):
        """FIXME! briefly describe function

        :param datax:
        :returns:
        :rtype:

        """

        X_pred = self.predict(datax).values

        return self.metrics_score(datax, X_pred)


"""____________________________________Main_________________________________"""
"""_________________________________________________________________________"""


def main():
    pass


if __name__ == "__main__":
    main()
