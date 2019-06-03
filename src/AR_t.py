#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

from IPython.display import HTML, display
from sklearn.base import clone

from __init__ import *
from AR import AR


class AR_t(AR):
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
        self.T = dict()

    @AR.datax_t_decorator
    def fit(self, datax, tplus):
        """ Finds the optimal weigths analytically

        :param datax: contient tous les exemples du dataset
        :returns: void
        :rtype: None

        """

        _, X, y = datax
        self.model = clone(self.initial_model)
        self.model.fit(X, y)

        return self

    @AR.datax_t_decorator
    def predict(self, datax, tplus):
        """

        """

        datax, X_test, y_test = datax

        yp = datax.iloc[:, :, 0].values.T.reshape(-1, 1)
        for t in range(1, self.p + tplus - 1):
            yp = np.vstack((
                yp,
                datax.iloc[:, :, t].values.T.reshape(-1, 1)))

        y_pred = self.model.predict(X_test).reshape(-1, 1)

        pred_matrix = self.reshaped(
            np.vstack((yp, y_pred)), datax, exact=True)

        self.T[tplus] = pd.Panel(pred_matrix,
                                 items=list(datax.items),
                                 major_axis=list(datax.major_axis),
                                 minor_axis=list(datax.minor_axis))

    def forecast(self, X_train, datax, tplus):
        """

        """

        for t in range(1, tplus + 1):
            self.fit(X_train, t)
            self.predict(datax, t)

        return self.T
