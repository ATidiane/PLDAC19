#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

from copy import deepcopy

from IPython.display import HTML, display
from sklearn.base import clone

from __init__ import *
from AR_s import AR_s


class AR_t_s(AR_s):
    def __init__(self, p, model):
        """FIXME! briefly describe function

        :param p:
        :param model:
        :returns:
        :rtype:

        """
        super().__init__(p, model)

    @AR_s.datax_t_decorator
    def fit(self, datax, tplus):
        """FIXME! briefly describe function

        :param datax:
        :returns:
        :rtype:

        """
        initial_datax, X_train, y_train = datax

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

    def forecast(self, X_train, datax, tplus=1):
        """FIXME! briefly describe function

        :param datax:
        :param tplus:
        :returns:
        :rtype:

        """

        TT = dict()
        for s in tqdm(range(datax.shape[1]), ascii=True, desc='Predicting'):
            X_station, X_test = self.decorate_X(datax.iloc[:, s].T)
            for t in range(1, tplus + 1):
                self.fit(X_train, t)
                self.predict(X_station, X_test, s, t)
            TT[s] = deepcopy(list(self.Ts.values()))
            self.Ts = dict()

        tmp = dict()
        for t in tqdm(range(tplus), ascii=True, desc='Creating T'):
            for s, i in zip(list(datax.major_axis), range(datax.shape[1])):
                tmp[s] = TT[i][t]

            self.T[t] = pd.Panel(tmp).transpose(1, 0, 2)

        return self.T
