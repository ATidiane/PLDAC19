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

    @Regressor.compute_residuals
    def compute_residuals(self, datax, y_pred_train):

        return datax

    @Regressor.X_y_decorator
    def fit(self, datax):

        _, self.X_train, self.y_train = datax
        self.model.fit(self.X_train, self.y_train.ravel())

    @Regressor.X_decorator
    def forecast(self, datax, tplus=1):
        datax, self.X_test = datax

        if len(self.T) == tplus - 1:

            if tplus > self.p + 1:
                a = tplus - self.p
                d = self.p + 1
            else:
                a, d = 1, tplus

            for i, j in zip(range(a, tplus),
                            reversed(range(1, d))):
                _, tmp = self.decorate_X(self.T[i])
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

    def score_t(self, datax, T):
        """FIXME! briefly describe function

        :param datax:
        :param T:
        :returns:
        :rtype:

        """

        scores = []
        for i, panel in enumerate(T):
            scores.append(self.metrics_score(
                datax.iloc[:, :, self.p + i:], panel.iloc[:, :, self.p + i:].values))

        display(HTML((pd.DataFrame(np.array(scores).T,
                                   index=['R2', 'RMSE', 'MSE', 'MAE', 'MAPE', 'MPE'],
                                   columns=list(map(
                                       lambda x: "t+" + str(x),
                                       range(1, len(scores) + 1))))).to_html()))

        return scores
