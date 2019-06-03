#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

from copy import deepcopy

from IPython.display import HTML, display
from sklearn.base import clone

from __init__ import *
from AR_sj import AR_sj
from constant import ddict_days


class AR_t_sj(AR_sj):
    def __init__(self, p, model, nb_days=7):
        """FIXME! briefly describe function

        :param p:
        :param model:
        :param nb_days:
        :returns:
        :rtype:

        """
        super().__init__(p, model)

    @AR_sj.datax_t_decorator
    def fit(self, datax, tplus):
        """FIXME! briefly describe function

        :param datax:
        :returns:
        :rtype:

        """
        initial_datax, X_train, y_train = datax

        for d in tqdm(range(self.nb_days), ascii=True, desc='Fitting'):
            exist_ind = list(
                set(ddict_days[d].values()) & set(initial_datax.items))

            initial_datax[exist_ind].apply(
                lambda station: self.models[d].append(
                    clone(self.initial_model).fit(*self.decorate_X_y(
                        station.T)[1:])),
                axis=(0, 2))

    def predict(self, X_station, X_test, d, s, tplus):
        """FIXME! briefly describe function

        :param X_station:
        :param X_test:
        :param d:
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

        y_pred = self.models[d][s].predict(X_test).reshape(-1, 1)

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

        for d in tqdm(range(self.nb_days), ascii=True, desc='Predicting'):
            exist_ind = list(
                set(ddict_days[d].values()) & set(datax.items))

            for s in range(datax.shape[1]):
                X_station, X_test = self.decorate_X(
                    datax[exist_ind].iloc[:, s].T)

                for t in range(1, tplus + 1):
                    self.fit(X_train, t)
                    self.predict(X_station, X_test, d, s, t)

                self.Tjs[d][s] = deepcopy(list(self.Ts.values()))
                self.Ts = dict()

        TT = {s: [] for s in range(datax.shape[1])}
        for t in tqdm(range(tplus), ascii=True, desc='Creating Ts'):
            for s in range(datax.shape[1]):
                tmp_s = self.Tjs[0][s][t]
                for d in range(1, self.nb_days):
                    tmp_s = pd.concat([tmp_s, self.Tjs[d][s][t]])

                TT[s].append(tmp_s.sort_index())

        tmp = dict()
        for t in tqdm(range(tplus), ascii=True, desc='Creating T'):
            for s, i in zip(list(datax.major_axis), range(datax.shape[1])):
                tmp[s] = TT[i][t]

            self.T[t] = pd.Panel(tmp).transpose(1, 0, 2)

        return self.T
