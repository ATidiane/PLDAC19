# -*- coding: utf-8 -*-

import numpy as np

from cost_functions import mse, mse_g


class myAR(object):
    def __init__(self, order=4, loss=mse, loss_g=mse_g, max_iter=1000,
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

    def forecast(self, datax):
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
                y_tplus1_pred = batchx.dot(self.w.T)
                y_pred[d].append(y_tplus1_pred)
                print(batchx)
                batchx = np.hstack(
                    (batchx[:, 1:], np.array(y_tplus1_pred).reshape(-1, 1)))

        return np.array(y_pred).transpose(0, 2, 1)

    def score(self, datax, datay):
        """ Evaluate de classification

        :param datax: contient les exemples du dataset
        :param datay: labels du dataset
        :returns: score des erreurs
        :rtype: float

        """

        return (1 - np.mean((self.predict(datax) == datay[:, np.newaxis])))
