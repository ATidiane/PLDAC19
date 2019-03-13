# -*- coding: utf-8 -*-

import numpy as np

from cost_functions import mse, mse_g

##########################################################################
# --------------------------------- Perceptron ------------------------- #
##########################################################################


class AR(object):
    def __init__(self, loss=mse, loss_g=mse_g, max_iter=1000,
                 eps=0.01):
        """ Initialisation des paramètres du perceptron

        :param loss: fonction de coût
        :param loss_g: gradient de la fonction coût
        :param max_iter: nombre maximum d'itération de la fonction coût
        :param eps: pas du gradient


        """

        self.max_iter, self.eps = max_iter, eps
        self.loss, self.loss_g = loss, loss_g

    def batch_fit(self, datax, datay):
        """ Classic gradient descent Learning

        :param datax: contient tous les exemples du dataset
        :param datay: labels du dataset

        """

        for i in range(self.max_iter):
            self.w = self.w - (self.eps * self.loss_g(datax, datay, self.w))

    def minibatch_fit(self, datax, datay, batch_size=10):
        """ Mini-Batch gradient descent Learning

        :param datax: contient tous les exemples du dataset
        :param datay: labels du dataset
        :param batch_size: nb d'exemples sur lesquels itérés en un.

        """

        for _ in range(self.max_iter):
            for i in range(0, datax.shape[0], batch_size):
                # On prend seulement batch_size données sur toutes les données.
                batchx, batchy = datax[i:i +
                                       batch_size], datay[i:i + batch_size]
                # Et on essaye de progresser avec cela.
                self.w -= (self.eps * self.loss_g(batchx, batchy, self.w))

    def predict(self, datax):
        """ Predict labels

        :param datax: contient tous les exemples du dataset
        :returns: predicted labels
        :rtype: numpy array

        """
        if len(datax.shape) == 1:
            datax = datax.reshape(-1, 1)

        return np.sign(datax.dot(self.w.T))

    def score(self, datax, datay):
        """ Evaluate de classification

        :param datax: contient les exemples du dataset
        :param datay: labels du dataset
        :returns: score des erreurs
        :rtype: float

        """

        return (1 - np.mean((self.predict(datax) == datay[:, np.newaxis])))
