#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

import numpy as np


class Regressor:
    def __init__(self):
        raise NotImplementedError("Please Implement this method")

    def datax_decorator(fonc):
        def reshape_data(self, datax, *args, **kwargs):
            """ Reshape data into one single matrix in order to apply analytic
            solution.

            :param datax: contient tous les exemples du dataset
            :returns: void
            :rtype: None

            """

            if datax.ndim == 3:
                X = datax.iloc[:, :,
                               0:self.order].values.reshape(-1, self.order)
                y = datax.iloc[:, :, self.order].values.T.reshape(-1, 1)
                for t in range(1, datax.shape[2] - self.order):
                    X = np.vstack((
                        X,
                        datax.iloc[:, :, t:t + self.order].values.reshape(-1, self.order)))
                    y = np.vstack((
                        y,
                        datax.iloc[:, :, t + self.order].values.T.reshape(-1, 1)))

                return fonc(self, (datax, X, y), *args, **kwargs)
            elif datax.ndim == 2:
                X = datax.iloc[:, 0:self.order].values.reshape(-1, self.order)
                y = datax.iloc[:, self.order].values.reshape(-1, 1)
                for t in range(1, datax.shape[1] - self.order):
                    X = np.vstack((
                        X,
                        datax.iloc[:, t:t + self.order].values.reshape(-1, self.order)))
                    y = np.vstack(
                        (y, datax.iloc[:, t + self.order].values.reshape(-1, 1)))

                return fonc(self, (datax, X, y), *args, **kwargs)
            elif datax.ndim == 1:
                # TODO
                pass
            else:
                raise ValueError("Untreated datax number of dimensions")

        return reshape_data

    def fit(self):
        raise NotImplementedError("Please Implement this method")

    def predict(self):
        raise NotImplementedError("Please Implement this method")

    def score(self):
        raise NotImplementedError("Please Implement this method")


"""____________________________________Main_________________________________"""
"""_________________________________________________________________________"""


def main():
    pass


if __name__ == "__main__":
    main()
