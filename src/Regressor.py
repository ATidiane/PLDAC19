#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

from __init__ import *


class Regressor:
    def __init__(self, p):
        self.p = p

    def X_y_decorator(fonc):
        def reshape_data(self, datax, *args, **kwargs):
            """ Reshape data into one single matrix in order to apply analytic
            solution.

            :param datax: contient tous les exemples du dataset
            :returns: void
            :rtype: None

            """

            if datax.ndim == 3:
                X = datax.iloc[:, :,
                               0:self.p].values.reshape(-1, self.p)
                y = datax.iloc[:, :, self.p].values.T.reshape(-1, 1)
                for t in range(1, datax.shape[2] - self.p):
                    X = np.vstack((
                        X,
                        datax.iloc[:, :, t:t + self.p].values.reshape(-1, self.p)))
                    y = np.vstack((
                        y,
                        datax.iloc[:, :, t + self.p].values.T.reshape(-1, 1)))

                return fonc(self, (datax, X, y), *args, **kwargs)
            elif datax.ndim == 2:
                X = datax.iloc[:, 0:self.p].values.reshape(-1, self.p)
                y = datax.iloc[:, self.p].values.reshape(-1, 1)
                for t in range(1, datax.shape[1] - self.p):
                    X = np.vstack((
                        X,
                        datax.iloc[:, t:t + self.p].values.reshape(-1, self.p)))
                    y = np.vstack(
                        (yb, datax.iloc[:, t + self.p].values.reshape(-1, 1)))

                return fonc(self, (datax, X, y), *args, **kwargs)
            elif datax.ndim == 1:
                # TODO
                pass
            else:
                raise ValueError("Untreated datax number of dimensions")

        return reshape_data

    def X_decorator(fonc):
        def reshape_data(self, datax, *args, **kwargs):
            """ Reshape data into one single matrix in order to apply analytic
            solution.

            :param datax: contient tous les exemples du dataset
            :returns: void
            :rtype: None

            """

            if datax.ndim == 3:
                X = datax.iloc[:, :,
                               0:self.p].values.reshape(-1, self.p)
                for t in range(1, datax.shape[2] - self.p):
                    X = np.vstack((
                        X,
                        datax.iloc[:, :, t:t + self.p].values.reshape(-1, self.p)))

                return fonc(self, (datax, X), *args, **kwargs)

            elif datax.ndim == 2:
                X = datax.iloc[:, 0:self.p].values.reshape(-1, self.p)

                for t in range(1, datax.shape[1] - self.p):
                    X = np.vstack((
                        X,
                        datax.iloc[:, t:t + self.p].values.reshape(-1, self.p)))

                return fonc(self, (datax, X), *args, **kwargs)

            elif datax.ndim == 1:
                # TODO
                pass
            else:
                raise ValueError("Untreated datax number of dimensions")

        return reshape_data

    def compute_residuals(fonc):
        def reshape_data(self, datax, y_pred_train, *args, **kwargs):
            if datax.ndim == 3:
                residuals = y_pred_train - datax.iloc[:, :, self.p:].values

                # Add zeros to the residuals and convert it into panel to match
                # datax
                zeros = np.zeros(
                    (residuals.shape[0], residuals.shape[1], self.p))
                residuals = np.concatenate((zeros, residuals), axis=2)
                pd_residuals = pd.Panel(residuals,
                                        items=list(datax.items),
                                        major_axis=list(datax.major_axis),
                                        minor_axis=list(datax.minor_axis))
                return fonc(self, pd_residuals, y_pred_train, *args, **kwargs)

            elif datax.ndim == 2:
                residuals = y_pred_train - datax.iloc[:, self.p:].values

                # Add zeros to the residuals and convert it into panel to match
                # datax
                zeros = np.zeros((residuals.shape[0], self.p))
                residuals = np.concatenate((zeros, residuals), axis=1)
                pd_residuals = pd.DataFrame(residuals,
                                            index=list(datax.index),
                                            columns=list(datax.columns))

                return fonc(self, pd_residuals, y_pred_train, *args, **kwargs)

            elif datax.ndim == 1:
                # TODO
                pass
            else:
                raise ValueError("Untreated datax number of dimensions")

        return reshape_data

    def datax_t_decorator(fonc):
        def reshape_data(self, datax, tplus, *args, **kwargs):
            """ Reshape data into one single matrix in order to apply analytic
            solution.

            :param datax: contient tous les exemples du dataset
            :returns: void
            :rtype: None

            """

            if datax.ndim == 3:
                X = datax.iloc[:, :,
                               0:self.p].values.reshape(-1, self.p)
                y = datax.iloc[:, :, self.p +
                               tplus - 1].values.T.reshape(-1, 1)
                for t in range(1, datax.shape[2] - self.p - tplus + 1):
                    X = np.vstack((
                        X,
                        datax.iloc[:, :, t:t + self.p].values.reshape(-1, self.p)))
                    y = np.vstack((
                        y,
                        datax.iloc[:, :, t + self.p + tplus - 1].values.T.reshape(-1, 1)))

                return fonc(self, (datax, X, y), tplus, *args, **kwargs)
            elif datax.ndim == 2:
                X = datax.iloc[:, 0:self.p].values.reshape(-1, self.p)
                y = datax.iloc[:, self.p + tplus - 1].values.reshape(-1, 1)
                for t in range(1, datax.shape[1] - self.p - tplus + 1):
                    X = np.vstack((
                        X,
                        datax.iloc[:, t:t + self.p].values.reshape(-1, self.p)))
                    y = np.vstack(
                        (y, datax.iloc[:, t + self.p + tplus - 1].values.reshape(-1, 1)))

                return fonc(self, (datax, X, y), tplus, *args, **kwargs)
            elif datax.ndim == 1:
                # TODO
                pass
            else:
                raise ValueError("Untreated datax number of dimensions")

        return reshape_data

    def reshaped(self, y_pred, datax, exact=False):
        """FIXME! briefly describe function

        :param y_pred:
        :param datax:
        :returns:
        :rtype:

        """
        if datax.ndim == 3:
            if exact:
                minor_axis = datax.shape[2]
            else:
                minor_axis = datax.shape[2] - self.p

            return y_pred.reshape(
                (datax.shape[0] *
                 datax.shape[1],
                 minor_axis),
                order='F').reshape(
                (datax.shape[0],
                 datax.shape[1],
                 minor_axis))
        elif datax.ndim == 2:
            if exact:
                minor_axis = datax.shape[1]
            else:
                minor_axis = datax.shape[1] - self.p
            return y_pred.reshape(
                (datax.shape[0], minor_axis), order='F')

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
