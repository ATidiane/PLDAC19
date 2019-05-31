#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

from copy import deepcopy

from IPython.display import HTML, display
from sklearn.base import clone

from __init__ import *
from AR import AR


class AR_s(AR):
    def __init__(self, p, model):
        """FIXME! briefly describe function

        :param p:
        :param model:
        :returns:
        :rtype:

        """
        super().__init__(p, model)
        self.p = p
        self.initial_model = model
        self.models = []
        self.T = dict()
        self.Ts = dict()

    @AR.X_y_decorator
    def fit(self, datax):
        """FIXME! briefly describe function

        :param datax:
        :returns:
        :rtype:

        """
        initial_datax, self.X_train, self.y_train = datax

        initial_datax.apply(lambda station: self.models.append(
            clone(self.initial_model).fit(*self.decorate_X_y(
                station.T)[1:])),
            axis=(0, 2))

    def predict(self, X_station, X_test, s, tplus):
        """FIXME! briefly describe function

        :param X_station:
        :param X_test:
        :param s:
        :param tplus:
        :returns:
        :rtype:

        """
        if len(self.Ts) == tplus - 1:
            for i, j in zip(range(1, tplus), reversed(range(1, tplus))):
                _, tmp = self.decorate_X(self.Ts[i])
                X_test[:, -j] = tmp[:, -j]

            yp = X_station.iloc[:, 0].values.reshape(-1, 1)
            for t in range(1, self.p):
                yp = np.vstack((
                    yp,
                    X_station.iloc[:, t].values.reshape(-1, 1)))

            y_pred = self.models[s].predict(X_test).reshape(-1, 1)

            pred_matrix = self.reshaped(
                np.vstack((yp, y_pred)), X_station, exact=True)

            self.Ts[tplus] = pd.DataFrame(pred_matrix,
                                          index=list(X_station.index),
                                          columns=list(X_station.columns))

        else:
            self.predict(X_station, X_test, s, tplus - 1)
            self.predict(X_station, X_test, s, tplus)

        return self.Ts

    @AR.X_decorator
    def forecast(self, datax, tplus=1):
        """FIXME! briefly describe function

        :param datax:
        :param tplus:
        :returns:
        :rtype:

        """
        datax, self.X_test = datax

        TT = dict()
        for s in tqdm(range(datax.shape[1]), ascii=True, desc='Predicting'):
            X_station, X_test = self.decorate_X(datax.iloc[:, s].T)
            self.predict(X_station, X_test, s, tplus)
            TT[s] = deepcopy(list(self.Ts.values()))
            self.Ts = dict()

        tmp = dict()
        for t in tqdm(range(tplus), ascii=True, desc='Creating T'):
            for s, i in zip(list(datax.major_axis), range(datax.shape[1])):
                tmp[s] = TT[i][t]

            self.T[t] = pd.Panel(tmp).transpose(1, 0, 2)

        return self.T
