import numpy as np

from cost_functions import mse, mse_g


class Regressor:
    def __init__(self):
        raise NotImplementedError("Please Implement this method")

    def datax_decorator(fonc):
        def reshape_data(self, datax, *args, **kwargs):
            """ Reshape data into one single matrix in order to apply analytic
            solution.

            :param datax: contient tous les exemples du dataset
            :returns: void
            :rtype: None

            """

            if datax.ndim == 3:
                X = datax.iloc[:, :,
                               0:self.order].values.reshape(-1, self.order)
                y = datax.iloc[:, :, self.order].values.T.reshape(-1, 1)
                for t in range(1, datax.shape[2] - self.order):
                    X = np.vstack((
                        X,
                        datax.iloc[:, :, t:t + self.order].values.reshape(-1, self.order)))
                    y = np.vstack((
                        y,
                        datax.iloc[:, :, t + self.order].values.T.reshape(-1, 1)))

                return fonc(self, (X, y), *args, **kwargs)
            elif datax.ndim == 2:
                X = datax.iloc[:, 0:self.order].values.reshape(-1, self.order)
                y = datax.iloc[:, self.order].values.reshape(-1, 1)
                for t in range(1, datax.shape[1] - self.order):
                    X = np.vstack((
                        X,
                        datax.iloc[:, t:t + self.order].values.reshape(-1, self.order)))
                    y = np.vstack((y,
                                   datax.iloc[:, t + self.order].values.reshape(-1, 1)))

                return fonc(self, (X, y), *args, **kwargs)
            elif datax.ndim == 1:
                # TODO
                pass
            else:
                raise ValueError("Untreated datax number of dimensions")

        return reshape_data

    def fit(self):
        raise NotImplementedError("Please Implement this method")

    def predict(self):
        raise NotImplementedError("Please Implement this method")

    def score(self):
        raise NotImplementedError("Please Implement this method")


class Baseline:
    def __init__(self, level=None, first_ndays=7):
        """ Initialization of Baseline parameters

        :param level: level of computing, means : if level is None the rmse is
        computed on the whole Train Set, if level is equal to "S", then the
        metric is applied separatly for each station, if it is equal to "J",
        it is applied separetly for each day : Lundi, Mardi, etc..., if it is
        equal to "SJ", then for each station and each day, we aggregate the
        values and compute the metric. And if level is none of the listed
        values above then, we consider that it is None, by default.

        :param first_ndays:
        :return:
        :rtype:

        """

        self.first_ndays = first_ndays
        self.level = level
        if self.level not in [None, "s", "S",
                              "j", "J", "sj", "SJ", "Sj", "sJ"]:
            self.level = None

    def fit(self, datax):
        """

        """

        if self.level is None:
            self.mean = datax.mean().mean(axis=1)
        elif self.level.lower() == "s":
            self.mean = datax.mean(axis=0)
        elif self.level.lower() == "j":
            self.mean = []
            for d in range(self.first_ndays):
                exist_ind = list(
                    set(ddict_days[d].values()) & set(datax.items))
                self.mean.append(datax[exist_ind].mean().mean(axis=1))
        elif self.level.lower() == "sj":
            self.mean = []
            for d in range(self.first_ndays):
                exist_ind = list(
                    set(ddict_days[d].values()) & set(datax.items))
                self.mean.append(datax[exist_ind].mean(axis=0))
        else:
            raise ValueError("Unknown value for level attribute, \
            try: s, j, sj or None")

    def predict(self, datax):
        """

        """

        if self.level is None:
            return datax.apply(lambda x: self.mean, axis="minor")

        elif self.level.lower() == "s":
            return datax.apply(lambda x: self.mean, axis=(1, 2))

        elif self.level.lower() == "j":
            datax_copy = datax.copy()
            for d in range(self.first_ndays):
                exist_ind = list(
                    set(ddict_days[d].values()) & set(datax.items))
                # print(exist_ind, "\n")
                datax_copy.update(datax_copy[exist_ind].apply(
                    lambda x: self.mean[d], axis="minor"))

            return datax_copy

        elif self.level.lower() == "sj":
            datax_copy = datax.copy()
            for d in range(self.first_ndays):
                exist_ind = list(
                    set(ddict_days[d].values()) & set(datax.items))
                datax_copy.update(datax_copy[exist_ind].apply(
                    lambda x: self.mean[d], axis=(1, 2)))

            return datax_copy

        else:
            raise ValueError("Unknown value for level attribute, \
            try: s, j, sj or None")

    def score(self, datax):
        """

        """

        return np.sqrt(((datax.values - self.predict(datax).values)**2).mean())


class myAR(Regressor):
    def __init__(self, order=4, level=None, loss=mse, loss_g=mse_g, max_iter=1000,
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

        self.X, self.y = datax
        A, B = self.X.T.dot(self.X), self.X.T.dot(self.y)
        self.w = np.linalg.solve(A, B).ravel()
        display(HTML(pd.DataFrame(self.w.reshape(1, -1), index=['Weights'],
                                  columns=range(len(self.w))).to_html()))

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

        # display(HTML(pd.DataFrame(self.w.reshape(1, -1), index=['Weights'],
        #                          columns=range(len(self.w))).to_html()))

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
                print(datax.to_frame())
                for d in range(datax.shape[0]):
                    y_pred.append([])
                    batchx = datax.iloc[d, :, 0:self.order].values
                    for t in range(0, datax.shape[2] - self.order):
                        print("======={}={}".format(d, t))
                        # print(batchx)
                        next_y = batchx.dot(self.w.T)
                        # next_y = np.where(next_y < 0, 0, next_y)
                        y_pred[d].append(next_y)
                        batchx = np.hstack(
                            (batchx[:, 1:], np.array(next_y).reshape(-1, 1)))
                        replace_ind = batch_ind + t + 1
                        batchx[:, batch_ind] = datax.iloc[d,
                                                          :, replace_ind].values
            elif datax.ndim == 2:
                # TODO
                pass
            elif datax.ndim == 1:
                batchx = datax.iloc[0:self.order].values
                # y_pred = batchx.tolist()
                for t in range(datax.shape[0] - self.order):
                    next_y = batchx.dot(self.w.T)
                    print(next_y)
                    if next_y < 0:
                        next_y = 0
                    y_pred.append(next_y)
                    batchx = np.hstack((batchx[1:], next_y))
                    replace_ind = replace_static_index + t + 1
                    batch_ind = self.order - tplus
                    batchx[batch_ind] = datax.iloc[replace_ind]

                return np.array(y_pred)
            else:
                raise ValueError("Untreated datax number of dimensions")

        return np.array(y_pred).transpose(0, 2, 1)
