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
from constant import ddict_days


class AR_j(AR):
    def __init__(self, p, model, nb_days=7):
        """FIXME! briefly describe function

        :param p:
        :param model:
        :param nb_days:
        :returns:
        :rtype:

        """
        super().__init__(p, model)
        self.p = p
        self.nb_days = nb_days
        self.initial_model = model
        self.models = []
        self.T = dict()
        self.Tj = dict()

    @AR.X_y_decorator
    def fit(self, datax):
        """FIXME! briefly describe function

        :param datax:
        :returns:
        :rtype:

        """
        initial_datax, self.X_train, self.y_train = datax

        for d in tqdm(range(self.nb_days), ascii=True, desc='Fitting'):
            exist_ind = list(
                set(ddict_days[d].values()) & set(initial_datax.items))

            self.models.append(clone(self.initial_model).fit(
                *self.decorate_X_y(initial_datax[exist_ind])[1:]))

    def predict(self, X_day, X_test, d, tplus):
        """FIXME! briefly describe function

        :param X_day:
        :param X_test:
        :param d:
        :param s:
        :param tplus:
        :returns:
        :rtype:

        """
        if len(self.Tj) == tplus - 1:
            for i, j in zip(range(1, tplus), reversed(range(1, tplus))):
                _, tmp = self.decorate_X(self.Tj[i])
                X_test[:, -j] = tmp[:, -j]

            yp = X_day.iloc[:, :, 0].values.T.reshape(-1, 1)
            for t in range(1, self.p):
                yp = np.vstack((
                    yp,
                    X_day.iloc[:, :, t].values.T.reshape(-1, 1)))

            y_pred = self.models[d].predict(X_test).reshape(-1, 1)

            pred_matrix = self.reshaped(
                np.vstack((yp, y_pred)), X_day, exact=True)

            self.Tj[tplus] = pd.Panel(pred_matrix,
                                      items=list(X_day.items),
                                      major_axis=list(X_day.major_axis),
                                      minor_axis=list(X_day.minor_axis))
        else:
            self.predict(X_day, X_test, d, tplus - 1)
            self.predict(X_day, X_test, d, tplus)

        return self.Tj

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
        for d in tqdm(range(self.nb_days), ascii=True, desc='Predicting'):
            exist_ind = list(
                set(ddict_days[d].values()) & set(datax.items))
            X_day, X_test = self.decorate_X(datax[exist_ind])
            self.predict(X_day, X_test, d, tplus)
            TT[d] = deepcopy(list(self.Tj.values()))
            self.Tj = dict()

        tmp = list()
        for t in tqdm(range(tplus), ascii=True, desc='Creating T'):
            for d in range(self.nb_days):
                tmp.append(TT[d][t])

            self.T[t] = pd.concat(deepcopy(tmp)).sort_index()
            tmp = list()

        return self.T
