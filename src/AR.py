#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

from IPython.display import HTML, display

from __init__ import *
from Regressor import Regressor


class AR(Regressor):
    def __init__(self, p, model):
        super().__init__(p)
        self.p = p
        self.initial_model = model
        self.model = model
        self.T = dict()

    @Regressor.X_y_decorator
    def decorate_X_y(self, datax):

        return datax

    @Regressor.X_decorator
    def decorate_X(self, datax):

        return datax

    @Regressor.X_y_decorator
    def fit(self, datax):

        _, self.X_train, self.y_train = datax
        self.model.fit(self.X_train, self.y_train)

    @Regressor.X_decorator
    def forecast(self, datax, tplus=1):
        datax, self.X_test = datax

        if len(self.T) == tplus - 1:
            for i, j in zip(range(1, tplus),
                            reversed(range(1, tplus))):
                _, tmp = self.decorate_X(self.T[i])
                self.X_test[:, -j] = tmp[:, -j]

            yp = datax.iloc[:, :, :self.p].values.reshape(-1, 1)
            y_pred = self.model.predict(self.X_test)

            pred_matrix = self.reshaped(
                np.vstack((yp, y_pred)), datax, exact=True)

            self.T[tplus] = pd.Panel(pred_matrix,
                                     items=list(datax.items),
                                     major_axis=list(datax.major_axis),
                                     minor_axis=list(datax.minor_axis))
        else:
            self.forecast(datax, tplus - 1)
            self.forecast(datax, tplus)
        return self.T

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

    def score(self, datax, T):
        """FIXME! briefly describe function

        :param datax:
        :param T:
        :returns:
        :rtype:

        """

        scores = []
        for panel in T:
            scores.append(self.metrics_score(
                datax.iloc[:, :, self.p:], panel.iloc[:, :, self.p:].values))

        display(HTML((pd.DataFrame(np.array(scores).T,
                                   index=['R2', 'RMSE', 'MSE', 'MAE', 'MAPE', 'MPE'],
                                   columns=list(map(
                                       lambda x: "t+" + str(x),
                                       range(1, len(scores) + 1))))).to_html()))

        return scores
