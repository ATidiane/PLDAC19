#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Ahmed Tidiane BALDE
"""

from IPython.display import HTML, display

from __init__ import *
from Baseline import Baseline
from utils import make_dict_intervals


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
    xv = (p - 1) * 900 + 4 * 3600
    ax[0].axvline(x=xv, linewidth=3, c='darkblue')
    for i in range(limit_t):
        pred_t = ar_preds[i].mean()
        pred_t.plot(ax=ax[i + 1], linewidth=2)
        ax[i + 1].axvline(x=xv, linewidth=3, c='brown')
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


def clustering_mape_map(
    contour_iris,
    station_data,
    dict_mape,
    intervals=[
        25,
        50]):
    """FIXME! briefly describe function

    :param contour_iris:
    :param station_data:
    :param dict_mape:
    :param intervals:
    :param 50]:
    :returns:
    :rtype:

    """
    dless_twf, dtwf_ffty, dmore_ffty = make_dict_intervals(
        dict_mape, intervals)

    kless_twf = list(dless_twf.keys())
    ktwf_ffty = list(dtwf_ffty.keys())
    kmore_ffty = list(dmore_ffty.keys())

    f, ax = plt.subplots(1, figsize=(16, 12))
    ax = contour_iris[contour_iris['dep'].isin([75, 92, 93, 94])].plot(
        ax=ax, edgecolor='darkgoldenrod', column='dep', cmap='binary_r')

    ax.scatter(station_data[station_data['id'].isin(kless_twf)]['x'],
               station_data[station_data['id'].isin(kless_twf)]['y'],
               color='lime',
               label='[0-{}]%'.format(intervals[0]))

    ax.scatter(station_data[station_data['id'].isin(ktwf_ffty)]['x'],
               station_data[station_data['id'].isin(ktwf_ffty)]['y'],
               color='yellow',
               label=']{0}-{1}]%'.format(intervals[0], intervals[1]))

    ax.scatter(station_data[station_data['id'].isin(kmore_ffty)]['x'],
               station_data[station_data['id'].isin(kmore_ffty)]['y'],
               color='red',
               label=']{0}-∞[%'.format(intervals[1]))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Clustering map of Subway Stations\'s MAPE in Paris')
    ax.legend()

    plt.show()


def clustering_mape_line(dict_mape, intervals):
    """FIXME! briefly describe function

    :param dict_mape:
    :param intervals:
    :returns:
    :rtype:

    """
    dless_twf, dtwf_ffty, dmore_ffty = make_dict_intervals(
        dict_mape, intervals=intervals)

    plt.figure(figsize=(16, 4))
    plt.axhline(y=25, linewidth=3, linestyle='--')
    plt.axhline(y=50, linewidth=3, linestyle='--')

    plt.plot(range(len(dless_twf)), sorted(list(dless_twf.values())),
             linewidth=3, c='lime', label='[0-{0}]%'.format(intervals[0]))
    plt.plot(range(len(dless_twf), len(dtwf_ffty) + len(dless_twf)),
             sorted(list(dtwf_ffty.values())), linewidth=3, c='yellow',
             label=']{0}-{1}]%'.format(intervals[0], intervals[1]))
    plt.plot(range(len(dless_twf) + len(dtwf_ffty),
                   len(dmore_ffty) + len(dless_twf) + len(dtwf_ffty)),
             sorted(list(dmore_ffty.values())), linewidth=3, c='red',
             label=']{0}-∞[%'.format(intervals[1]))
    plt.legend()


def ar_against_baseline(ar_scores, baseline_score, labels, title):
    """FIXME! briefly describe function

    :param ar_scores:
    :param df_baseline_scores:
    :param labels:
    :param title:
    :returns:
    :rtype:

    """
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.set_prop_cycle(color=['darkgoldenrod', 'brown'])
    ar_limit_t = len(ar_scores)
    x = range(1, ar_limit_t + 1)
    baseline_score = baseline_score.repeat(
        ar_limit_t).reshape(-1, ar_limit_t).T
    model_score = np.array(ar_scores).T[1]

    ax = plt.plot(x, model_score, linewidth=3, label=labels[0])
    ax = plt.scatter(x, model_score, marker='*', s=100)
    ax = plt.plot(x, baseline_score, linewidth=3, label=labels[1])
    ax = plt.scatter(x, baseline_score, marker='*', s=100)

    plt.legend(prop={'size': 20})
    plt.title(title + ", from $t+1$ to $t+{}$".format(ar_limit_t), fontsize=16)
    plt.xlabel("T plus", fontsize=16)
    plt.ylabel("RMSE", fontsize=16)
