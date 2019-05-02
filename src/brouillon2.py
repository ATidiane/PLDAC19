from sklearn.linear_model import Lasso, LinearRegression

from cost_functions import mse, mse_g


class myAR(Regressor):
    def __init__(
            self,
            order=4,
            level=None,
            loss=mse,
            loss_g=mse_g,
            max_iter=1000,
            eps=0.01):
        """ Initialisation des paramètres du perceptron

        :param order: Taille de la fenêtre glissante
        :param loss: fonction de coût
        :param loss_g: gradient de la fonction coût
        :param max_iter: nombre maximum d'itération de la fonction coût
        :param eps: pas du gradient


        """

        self.order = order
        self.max_iter, self.eps = max_iter, eps
        self.loss, self.loss_g = loss, loss_g
        self.w = np.random.random(self.order)

    @Regressor.datax_decorator
    def analytic_fit(self, datax):
        """ Finds the optimal weigths analytically

        :param datax: contient tous les exemples du dataset
        :returns: void
        :rtype: None

        """

        self.reg = LinearRegression()
        _, self.X, self.y = datax
        A, B = self.X.T.dot(self.X), self.X.T.dot(self.y)
        self.w1 = np.linalg.solve(A, B).ravel()
        self.reg.fit(self.X, self.y)
        self.w = self.reg.coef_.squeeze()
        display(HTML(pd.DataFrame(self.w.reshape(1, -1), index=['Weights'],
                                  columns=range(1, len(self.w) + 1)).to_html()))

        return self

    def minibatch_fit(self, datax):
        """ Mini-Batch gradient descent Learning

        :param datax: contient tous les exemples du dataset

        """

        for _ in range(self.max_iter):
            for d in range(datax.shape[0]):
                for t in range(datax.shape[2] - self.order):
                    batchx = datax.iloc[d, :, t:t + self.order].values
                    batchy = datax.iloc[d, :, t + self.order].values
                    self.w -= (self.eps * self.loss_g(batchx, batchy, self.w))

    def predict(self, datax):
        """ Predict labels

        :param datax: contient tous les exemples du dataset
        :returns: predicted labels
        :rtype: numpy array

        """

        y_pred = []
        for d in range(datax.shape[0]):
            y_pred.append([])
            for t in range(datax.shape[2] - self.order):
                batchx = datax.iloc[d, :, t:t + self.order].values
                y_pred[d].append(batchx.dot(self.w.T))

        return np.array(y_pred).transpose(0, 2, 1)

    def forecast_n(self, datax):
        """ Predict labels

        :param datax: contient tous les exemples du dataset
        :returns: predicted labels
        :rtype: numpy array

        """

        y_pred = []
        for d in range(datax.shape[0]):
            y_pred.append([])
            batchx = datax.iloc[d, :, 0:self.order].values
            for t in range(datax.shape[2] - self.order):
                next_y = batchx.dot(self.w.T)
                y_pred[d].append(next_y)
                batchx = np.hstack(
                    (batchx[:, 1:], np.array(next_y).reshape(-1, 1)))

        return np.array(y_pred).transpose(0, 2, 1)

    def transform_batchx(self, batchx, tplus):
        """
        """
        if tplus == 1:
            return batchx

        for _ in range(tplus - 1):
            next_y = batchx.dot(self.w.T)
            if batchx.ndim == 2:
                batchx = np.hstack((batchx[:, 1:],
                                    np.array(next_y).reshape(-1, 1)))
            elif batchx.ndim == 1:
                batchx = np.hstack((batchx[1:], next_y))

        return batchx

    def forecast(self, datax, tplus=None):
        """ Predict labels

        :param datax: contient tous les exemples du dataset
        :param tplus: if t equal to 2, means predicting what happened at t+2
        :returns: predicted labels
        :rtype: numpy array

        """

        if tplus is None or tplus > self.order:
            return self.forecast_n(datax)
        else:
            y_pred = []
            batch_ind = self.order - tplus

            if datax.ndim == 3:
                for d in range(datax.shape[0]):
                    y_pred.append([])
                    # Take the first batch
                    batchx = datax.iloc[d, :, 0:self.order].values
                    # Predict till we finish the first round of tplus
                    for _ in range(tplus):
                        next_y = batchx.dot(self.w.T)
                        y_pred[d].append(next_y)
                        batchx = np.hstack((batchx[:, 1:],
                                            np.array(next_y).reshape(-1, 1)))

                    # After the first round of tplus, we have to replace some
                    # predicted values by the real ones and simultaneously
                    # replace the following columns by t+1,..., tplus
                    for t in range(1, datax.shape[2] - self.order - tplus + 1):
                        batchx = self.transform_batchx(
                            datax.iloc[d, :, t:self.order + t].values, tplus)
                        next_y = batchx.dot(self.w.T)
                        # next_y = np.where(next_y < 0, 0, next_y)
                        y_pred[d].append(next_y)
            elif datax.ndim == 2:
                # TODO
                pass
            elif datax.ndim == 1:
                batchx = datax.iloc[0:self.order].values

                for _ in range(tplus):
                    next_y = batchx.dot(self.w.T)
                    y_pred.append(next_y)
                    batchx = np.hstack((batchx[1:], next_y))

                print(datax.shape[0])
                for t in range(1, datax.shape[0] - self.order - tplus + 1):
                    batchx = self.transform_batchx(
                        datax.iloc[t:self.order + t].values, tplus)
                    next_y = batchx.dot(self.w.T)
                    # if next_y < 0: next_y = 0
                    y_pred.append(next_y)

                return np.array(y_pred)
            else:
                raise ValueError("Untreated datax number of dimensions")

        return np.array(y_pred).transpose(0, 2, 1)

    @Regressor.datax_decorator
    def again(self, datax):
        datax, self.X_test, self.y_test = datax

        y_pred = self.reg.predict(self.X_test)
        y_pred = y_pred.reshape(
            (datax.shape[0] *
             datax.shape[1],
             datax.shape[2] -
             self.order),
            order='F').reshape(
            (datax.shape[0],
             datax.shape[1],
             datax.shape[2] -
             self.order))

        return y_pred

    @Regressor.datax_decorator
    def again_tplus(self, datax, tplus):
        datax, self.X_test, self.y_test = datax
        self.X_test = self.X_test.reshape(
            datax.shape[-1] - self.order, -1, self.order)

        tmp = self.X_test[0]
        y_pred = self.reg.predict(tmp)
        pred = y_pred.copy()
        for x in self.X_test[1:]:
            x[:, -1] = pred.squeeze()
            x[:, -tplus:-1] = tmp[:, -tplus + 1:]
            tmp = x.copy()
            pred = self.reg.predict(tmp)
            pred = np.where(pred < 0, , pred)
            y_pred = np.vstack((y_pred, pred))

        print(y_pred)
        y_pred = y_pred.reshape(
            (datax.shape[0] *
             datax.shape[1],
             datax.shape[2] -
             self.order),
            order='F').reshape(
            (datax.shape[0],
             datax.shape[1],
             datax.shape[2] -
             self.order))

        return y_pred

    @Regressor.datax_decorator
    def again_s(self, datax):
        datax, self.X_test, self.y_test = datax

        y_pred = self.reg.predict(self.X_test)
        y_pred = y_pred.reshape((datax.shape[0], datax.shape[1] - self.order),
                                order='F')

        return y_pred

    class theAR(Baseline):
    station_id = 0

    def __init__(self, level=None, first_ndays=7, **kwargs):
        """

        """

        super().__init__(level, first_ndays)
        self.kwargs = kwargs

    def fit(self, datax):
        """

        """

        if self.level is None:
            self.model = myAR(**self.kwargs)
            self.model.analytic_fit(datax)

        elif self.level.lower() == "s":

            self.models = []
            # for s in range(datax.shape[1]):
            #    Xs = datax.iloc[:, s].T
            #    self.models.append(myAR(**self.kwargs))
            #    self.models[s].analytic_fit(Xs)

            datax.apply(lambda station: self.models.append(
                myAR(**self.kwargs).analytic_fit(station.T)),
                axis=(0, 2))

        elif self.level.lower() == "j":
            # TODO
            self.mean = []
            for d in range(self.first_ndays):
                exist_ind = list(
                    set(ddict_days[d].values()) & set(datax.items))
                self.mean.append(datax[exist_ind].mean().mean(axis=1))

        elif self.level.lower() == "sj":
            # TODO
            self.mean = []
            for d in range(self.first_ndays):
                exist_ind = list(
                    set(ddict_days[d].values()) & set(datax.items))
                self.mean.append(datax[exist_ind].mean(axis=0))
        else:
            raise ValueError("Unknown value for level attribute, \
            try: s, j, sj or None")

    def predict(self, datax, tplus=None):
        """

        """

        def predict_per_station(x, tplus):
            pred_s = []
            for s in range(x.shape[0]):
                pred_s.append(self.models[s].forecast(x.iloc[s], tplus))
            return np.array(pred_s)

        def predict_for_station(x):
            """
            """

            station_pred = self.models[self.station_id].again_s(x)
            self.station_id += 1

            return station_pred

        if self.level is None:

            X_pred = self.model.again_tplus(datax)

            self.scores = super().metrics_score(
                datax.iloc[:, :, self.model.order:], X_pred)

            return X_pred

        elif self.level.lower() == "s":

            # return datax.apply(
            #    lambda x: predict_per_station(x, tplus), axis=(1, 2))

            X_pred = datax.apply(lambda x: predict_for_station(x.T),
                                 axis=(0, 2)).transpose(1, 0, 2)

            self.scores = super().metrics_score(
                datax.iloc[:, :, self.models[0].order:], X_pred.values)

            return X_pred

        elif self.level.lower() == "j":
            # TODO
            pass
        elif self.level.lower() == "sj":
            # TODO
            pass
        else:
            raise ValueError("Unknown value for level attribute, \
            try: s, j, sj or None")

    def score(self, datax, tplus=None):
        """

        """

        return self.scores

    def ar_plot_results(level, order, limit_t):
    """

    """

    ar_scores = []
    ar_preds = []
    ar = theAR(level=level, order=order)
    print("Fitting...")
    ar.fit(X_train)
    print("Predicting...")

    # for t in tqdm(range(1, limit_t+1)):
    #    ar_preds.append(ar.predict(X_test, t))
    #    ar_scores.append(ar.score(X_test, t))

    # print(ar_preds[0])

    # return ar_scores, ar_preds

    ar_preds = ar.predict(X_test, limit_t print("Xtest", self.X_test))
    print("Scoring...")
    ar_scores = [ar.score(X_test, 2)]
    print(ar_scores)
    display(HTML((pd.DataFrame(np.array(ar_scores).T,
                               index=['R2', 'RMSE', 'MSE', 'MAE', 'MAPE', 'MPE'],
                               columns=list(map(
                                   lambda x: "t+" + str(x),
                                   range(1, len(ar_scores) + 1))))).to_html()))

    return [ar_preds]


def plot_qualitative_analysis(*args):
    """

    """

    fig, ax = plt.subplots(limit_t + 1, figsize=(16, limit_t * 4))

    wd_testorder_15m = args[1].iloc[:, :, order:]
    wdm_testorder_15m = wd_testorder_15m.mean()

    wdm_testorder_15m.plot(ax=ax[0])
    ax[0].set_ylabel('Number of Validations')
    ax[0].set_title('Test')
    ax[0].legend(bbox_to_anchor=(1., 0.9, 1.1, .102), ncol=2, loc=2,
                 borderaxespad=0.)

    for i in range(1, limit_t + 1):
        pred_t = create_panel_pred(*args).mean()
        pred_t.plot(ax=ax[i])
        ax[i].set_ylabel('Number of Validations')
        ax[i].set_title("Predict t+{}".format(i))
        ax[i].legend(bbox_to_anchor=(1., 0.9, 1.1, .102), ncol=2, loc=2,
                     borderaxespad=0.)

    plt.tight_layout()
    plt.show()
