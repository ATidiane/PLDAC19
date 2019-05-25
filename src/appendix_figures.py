#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

from IPython.display import HTML, display

from __init__ import *
from Baseline import Baseline


def baseline_plot_results(levels, X_train, X_test, first_ndays=7):
    """

    :param levels:
    :param X_train:
    :param X_test:
    :param first_ndays:

    """

    baseline_scores = []
    baseline_preds = []
    for level in levels:
        b = Baseline(level=level, first_ndays=first_ndays)
        b.fit(X_train)
        baseline_preds.append(b.predict(X_test))
        baseline_scores.append(b.score(X_test))

    df_baseline_scores = pd.DataFrame(
        np.array(baseline_scores).T,
        index=[
            'R2',
            'RMSE',
            'MSE',
            'MAE',
            'MAPE',
            'MPE'],
        columns=levels)
    display(HTML(df_baseline_scores.to_html()))

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
    tmp = pd.DataFrame(df_baseline_scores.loc['RMSE'].values.repeat(
        4).reshape(-1, 4).T, columns=levels)
    tmp.plot(ax=ax[0], kind='line')
    tmp.iloc[0].plot(ax=ax[1], kind='bar')

    return df_baseline_scores, baseline_preds


def plot_qualitative_analysis(X_test, ar_preds, p=4):
    """

    :param X_test:
    :param ar_preds:
    :param p:

    """
    limit_t = len(ar_preds)
    fig, ax = plt.subplots(limit_t + 1, figsize=(16, (limit_t + 1) * 3))

    Xm = X_test.mean()

    Xm.plot(ax=ax[0], linewidth=2)
    ax[0].set_ylabel('Number of Validations')
    ax[0].set_title('Test')
    ax[0].legend(bbox_to_anchor=(1., 0.9, 1.1, .102), ncol=2, loc=2,
                 borderaxespad=0.)
    # Convert del_hours + p hours into second
    xv = p * 900 + 4 * 3600
    ax[0].axvline(x=xv, linewidth=5, c='darkblue')
    for i in range(limit_t):
        pred_t = ar_preds[i].mean()
        pred_t.plot(ax=ax[i + 1], linewidth=2)
        ax[i + 1].axvline(x=xv, linewidth=5, c='brown')
        ax[i + 1].set_ylabel('Number of Validations')
        ax[i + 1].set_title("Predict t+{}".format(i + 1))
        ax[i + 1].legend(bbox_to_anchor=(1., 0.9, 1.1, .102), ncol=2, loc=2,
                         borderaxespad=0.)

    plt.tight_layout()
    plt.show()


def plot_specific(X_test, ar_preds, ar_preds_s, baseline_preds, p, j, s):
    """FIXME! briefly describe function

    :param X_test:
    :param ar_preds:
    :param ar_preds_s:
    :param baseline_preds:
    :param p:
    :param j:
    :param s:
    :returns:
    :rtype:

    """
    limit_t = len(ar_preds)
    fig, ax = plt.subplots(limit_t, figsize=(16, limit_t * 5))

    for t in range(limit_t):
        ar_preds[t].iloc[j, s].plot(ax=ax[t], label='General AR')
        ar_preds_s[t].iloc[j, s].plot(ax=ax[t], label='AR By Station')
        X_test.iloc[j, s].plot(ax=ax[t], label="Real values")
        baseline_preds[0].iloc[j, s].plot(
            ax=ax[t], style=['.--'], label='General Baseline')
        baseline_preds[1].iloc[j, s].plot(
            ax=ax[t], style=['.--'], label='Baseline per station')
        ax[t].set_ylabel('Number of Validations')
        ax[t].set_title(
            "AR models at t+{} with an order of {}".format(t + 1, p))
        ax[t].legend(bbox_to_anchor=(1., 0.9, 1.1, .102), ncol=1, loc=2,
                     borderaxespad=0.)

    plt.tight_layout()
    plt.show()


def plot_bispecific(X_test, ar_preds, ar_preds_s, baseline_preds, j, s, p=4):
    """FIXME! briefly describe function

    :param X_test:
    :param ar_preds:
    :param ar_preds_s:
    :param baseline_preds:
    :param j:
    :param s:
    :param p:
    :returns:
    :rtype:

    """

    limit_t = len(ar_preds)
    fig, ax = plt.subplots(limit_t, 2, figsize=(16, limit_t * 5))

    for t in range(limit_t):
        ar_preds[t].iloc[j, s].plot(ax=ax[t][0], label='General AR')
        ar_preds_s[t].iloc[j, s].plot(ax=ax[t][1], label='AR By Station')
        for c in range(2):
            X_test.iloc[j, s].plot(ax=ax[t][c], label="Real values")
            baseline_preds[0].iloc[j, s].plot(
                ax=ax[t][c], style=['.--'], label='General Baseline')
            baseline_preds[1].iloc[j, s].plot(
                ax=ax[t][c], style=['.--'], label='Baseline per station')
            ax[t][c].set_ylabel('Number of Validations')
            ax[t][c].set_title(
                "AR models at t+{} with an order of {}".format(t + 1, p))
            ax[t][c].legend(bbox_to_anchor=(0.68, 0.9, 1.1, .102),
                            ncol=1, loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.show()


def plot_diff_along_time(X_test, ar_preds):
    """FIXME! briefly describe function

    :param X_test:
    :param ar_preds:
    :param p:
    :returns:
    :rtype:

    """
    res = []
    for t in range(len(ar_preds)):
        res.append(np.sqrt(((X_test.values -
                             ar_preds[t].values)**2).mean(axis=(0, 1))))

    pd_res = pd.DataFrame(np.array(res).T, index=list(ar_preds[t].minor_axis),
                          columns=list(map(lambda x: "t+" + str(x),
                                           range(1, len(ar_preds) + 1))))
    pd_res.plot(
        figsize=(16, 6),
        title='Plot of RMSE of different predictions along days')
