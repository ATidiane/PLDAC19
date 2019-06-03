#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""
from IPython.display import HTML, display

from __init__ import *
from AR import AR


class ARMA(AR):
    def __init__(self, p, q, model):
        super().__init__(p, model)
        self.q = q
        # q cannot be greater that p
        if self.q > self.p:
            self.q = self.p

    @AR.X_y_decorator
    def fit(self, datax):

        datax, self.X_train, self.y_train = datax

        ar = AR(self.p, self.model)
        ar.fit(datax)
        y_pred_train = ar.forecast(datax)[1].iloc[:, :, self.p:].values

        pd_residuals = self.compute_residuals(datax, y_pred_train)

        # Reshaping X_err_train to match X_train
        residuals, X_err_train, y_err_train = self.decorate_X_y(pd_residuals)

        # Add self.q errors to X_train
        self.X_train = self.X_train if self.q == 0 else np.concatenate(
            (self.X_train, X_err_train[:, :self.q]), axis=1)

        # Fit new training Data with residuals
        self.model.fit(self.X_train, self.y_train.ravel())

    def forecast(self, X_train, datax, tplus=1):
        datax, self.X_test = super().decorate_X(datax)

        self.fit(X_train)

        zeros = np.zeros((self.X_test.shape[0], self.q))
        self.X_test = np.concatenate((self.X_test, zeros), axis=1)

        if len(self.T) == tplus - 1:
            if tplus > self.p + 1:
                a = tplus - self.p
                d = self.p + 1 + self.q
            else:
                a, d = 1, tplus + self.q

            for i, j in zip(range(a, tplus),
                            reversed(range(1 + self.q, d))):
                _, tmp = self.decorate_X(self.T[i])
                tmp = np.concatenate((tmp, zeros), axis=1)
                self.X_test[:, -j] = tmp[:, -j]

            yp = datax.iloc[:, :, 0].values.T.reshape(-1, 1)
            for t in range(1, self.p):
                yp = np.vstack((
                    yp,
                    datax.iloc[:, :, t].values.T.reshape(-1, 1)))

            y_pred = self.model.predict(self.X_test).reshape(-1, 1)

            pred_matrix = self.reshaped(
                np.vstack((yp, y_pred)), datax, exact=True)

            self.T[tplus] = pd.Panel(pred_matrix,
                                     items=list(datax.items),
                                     major_axis=list(datax.major_axis),
                                     minor_axis=list(datax.minor_axis))
        else:
            self.forecast(X_train, datax, tplus - 1)
            self.forecast(X_train, datax, tplus)
        return self.T
