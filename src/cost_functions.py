# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

import numpy as np

"""______________________________Cost Functions_____________________________"""
"""_________________________________________________________________________"""


def decorator_vec(fonc):
    def vecfonc(datax, datay, w, *args, **kwargs):
        """ decorator

        :param datax: contient tous les exemples du dataset
        :param datay: labels du dataset
        :param w: vecteur poids
        :returns: modifies datax, datay and w
        :rtype: numpy arrays

        """

        if not hasattr(datay, "__len__"):
            datay = np.array([datay])
        datax, datay, w = datax.reshape(
            len(datay), -1), datay.reshape(-1, 1), w.reshape((1, -1))
        return fonc(datax, datay, w, *args, **kwargs)
    return vecfonc


@decorator_vec
def mse(datax, datay, w):
    """ retourne la moyenne de l'erreur aux moindres carres

    :param datax: contient tous les exemples du dataset
    :param datay: labels du dataset
    :param w: vecteur poids

    """

    # (w.x - y)² np.mean(((datax * w) - datay)**2)
    return np.mean((np.dot(datax, w.T) - datay)**2)


@decorator_vec
def mse_g(datax, datay, w):
    """ retourne le gradient moyen de l'erreur au moindres carres

    :param datax: contient tous les exemples du dataset
    :param datay: labels du dataset
    :param w: vecteur poids

    """

    # 2 x (w.x - y)/m
    return np.mean(2 * (np.dot(datax, w.T) - datay))


@decorator_vec
def hinge(datax, datay, w):
    """ retourn la moyenne de l'erreur hinge

    :param datax: contient tous les exemples du dataset
    :param datay: labels du dataset
    :param w: vecteur poids

    """

    # Erreur moyenne de -f(x)y
    moinsyfx = -datay * np.dot(datax, w.T)
    return np.maximum(0, moinsyfx).mean()


@decorator_vec
def hinge_g(datax, datay, w, activation=np.sign):
    """ Retourne le gradient de l'erreur hinge

    :param datax: contient tous les exemples du dataset
    :param datay: labels du dataset
    :param w: vecteur poids
    :param activation:

    """

    cost = -activation(hinge(datax, datay, w)) * datax * datay
    return (np.sum(cost, axis=0) / len(datax))  # Normalisation


def stochastic(vectorx, vectory, w):
    """ Retourne l'erreur aux moindres carres pour UN exemple de data,
        cette pratique de calculer la descente de gradient juste pour un ex
        à chaque itération est appelée descente de gradient stochastique ou
        Stochastic gradient descent.

    :param vectorx: un exemple de la base
    :param vectory: label d'un exemple de la base
    :param w: vecteur poids
    :param activation:

    """

    return ((np.dot(vectorx.reshape(1, -1), w.T) - vectory)**2) / 2


def stochastic_g(vectorx, vectory, w):
    """ Retourne le gradient de l'erreur aux moindres carres pour UN exemple de
        datax.

    :param vectorx: un exemple de la base
    :param vectory: label d'un exemple de la base
    :param w: vecteur poids
    """
    return (np.dot(vectorx.reshape(1, -1), w.T) - vectory)


"""____________________________________Main_________________________________"""
"""_________________________________________________________________________"""


def main():
    pass


if __name__ == '__main__':
    main()
