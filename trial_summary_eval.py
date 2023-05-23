import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed

colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']
female_color, male_color = '#e74c3c', '#3498db'
Wc, Lc = 'darkgreen', '#3673A4'


def plot_rise_vs_chirp_count(trial_summary):
    fig = plt.figure(figsize=(20/2.54, 20/2.54))
    gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, height_ratios=[1, 3], width_ratios=[3, 1])
    ax = fig.add_subplot(gs[1, 0])

    ax.plot(trial_summary['rises_win'], trial_summary['chirps_win'], 'o', color=Wc, label='winner')
    ax.plot(trial_summary['rises_lose'], trial_summary['chirps_lose'], 'o', color=Lc, label='loster')
    ax.set_xlabel('rises [n]', fontsize=12)
    ax.set_ylabel('chirps [n]', fontsize=12)
    ax.tick_params(labelsize=10)

    ax_chirps = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_chirps.boxplot([trial_summary['chirps_win'], trial_summary['chirps_lose']], widths = .5, positions = [1, 2])
    ax_chirps.set_xticks([1, 2])
    ax_chirps.set_xticklabels(['Win', 'Lose'])
    plt.setp(ax_chirps.get_yticklabels(), visible=False)

    ax_rises = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_rises.boxplot([trial_summary['rises_win'], trial_summary['rises_lose']], widths = .5, positions = [1, 2], vert=False)
    ax_rises.set_yticks([1, 2])
    ax_rises.set_yticklabels(['Win', 'Lose'])
    plt.setp(ax_rises.get_xticklabels(), visible=False)


def plot_beh_count_per_pairing(trial_summary,
                               beh_key_win=None, beh_key_lose=None,
                               ylabel='y'):

    mek = ['k', 'None', 'None', 'k']
    markersize = 12
    win_colors = [male_color, male_color, female_color, female_color]
    lose_colors = [male_color, female_color, male_color, female_color]

    win_count = []
    lose_count = []

    for win_sex, lose_sex in itertools.product(['m', 'f'], repeat=2):
        win_count.append(trial_summary[beh_key_win][(trial_summary["sex_win"] == win_sex) &
                                                    (trial_summary["sex_lose"] == lose_sex) &
                                                    (trial_summary["draw"] == 0)].to_numpy())
        lose_count.append(trial_summary[beh_key_lose][(trial_summary["sex_win"] == win_sex) &
                                                      (trial_summary["sex_lose"] == lose_sex) &
                                                      (trial_summary["draw"] == 0)].to_numpy())

    fig = plt.figure(figsize=(20/2.54, 12/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
    ax = fig.add_subplot(gs[0, 0])
    ax.boxplot(win_count, positions=np.arange(len(win_count))-0.15, widths= .2, sym='')
    ax.boxplot(lose_count, positions=np.arange(len(lose_count))+0.15, widths= .2, sym='')

    ax.set_xticks(np.arange(len(win_count)))
    ax.set_xticklabels([u'\u2642\u2642', u'\u2642\u2640', u'\u2640\u2642', u'\u2640\u2640'])
    # ax.set_xticklabels(['mm', 'mf', 'fm', 'ff'])
    y0, y1 = ax.get_ylim()
    for i in range(len(win_count)):
        ax.text(i, y1, f'n={len(win_count[i]):.0f}', fontsize=10, ha='center', va='bottom')
    ax.set_ylim(top = y1*1.1)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.tick_params(labelsize=10)


def plot_beh_count_vs_meta(trial_summary,
                           beh_key_win=None, beh_key_lose=None,
                           meta_key_win=None, meta_key_lose=None,
                           xlabel='x'):
    mek = ['k', 'None', 'None', 'k']
    markersize = 12
    win_colors = [male_color, male_color, female_color, female_color]
    lose_colors = [male_color, female_color, male_color, female_color]


    win_count = []
    lose_count = []

    win_meta = []
    lose_meta = []

    for win_sex, lose_sex in itertools.product(['m', 'f'], repeat=2):
        win_count.append(trial_summary[beh_key_win][(trial_summary["sex_win"] == win_sex) &
                                                      (trial_summary["sex_lose"] == lose_sex) &
                                                      (trial_summary["draw"] == 0)].to_numpy())
        lose_count.append(trial_summary[beh_key_lose][(trial_summary["sex_win"] == win_sex) &
                                                        (trial_summary["sex_lose"] == lose_sex) &
                                                        (trial_summary["draw"] == 0)].to_numpy())

        win_meta.append(trial_summary[meta_key_win][(trial_summary["sex_win"] == win_sex) &
                                           (trial_summary["sex_lose"] == lose_sex) &
                                           (trial_summary["draw"] == 0)].to_numpy())
        lose_meta.append(trial_summary[meta_key_lose][(trial_summary["sex_win"] == win_sex) &
                                            (trial_summary["sex_lose"] == lose_sex) &
                                            (trial_summary["draw"] == 0)].to_numpy())

    fig = plt.figure(figsize=(20/2.54, 20/2.54))
    gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0]))
    ax.append(fig.add_subplot(gs[0, 1], sharey=ax[0]))
    ax.append(fig.add_subplot(gs[1, 1], sharex=ax[2], sharey=ax[1]))

    for i in range(len(win_count)):
        ax[0].plot(win_meta[i]-lose_meta[i], win_count[i], 'p', color=win_colors[i], markeredgecolor=mek[i], markersize=markersize, markeredgewidth=2)
        ax[1].plot(win_meta[i]-lose_meta[i], lose_count[i], 'p', color=win_colors[i], markeredgecolor=mek[i], markersize=markersize, markeredgewidth=2)

        ax[2].plot((win_meta[i]-lose_meta[i])*-1, win_count[i], 'o', color=lose_colors[i], markeredgecolor=mek[i], markersize=markersize, markeredgewidth=2)
        ax[3].plot((win_meta[i]-lose_meta[i])*-1, lose_count[i], 'o', color=lose_colors[i], markeredgecolor=mek[i], markersize=markersize, markeredgewidth=2   )

    ax[0].set_ylabel(f'{beh_key_win} [n]', fontsize=12)
    ax[1].set_ylabel(f'{beh_key_lose} [n]', fontsize=12)
    ax[1].set_xlabel(f'{xlabel}', fontsize=12)
    ax[3].set_xlabel(f'{xlabel}', fontsize=12)

    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[2].get_xticklabels(), visible=False)

    plt.setp(ax[2].get_yticklabels(), visible=False)
    plt.setp(ax[3].get_yticklabels(), visible=False)
    plt.tick_params(labelsize=10)


def plot_beh_conut_vs_experience(trial_summary, beh_key_win='chirps_win', beh_key_lose='chirps_lose', ylabel='chirps [n]'):
    mek = ['k', 'None', 'None', 'k']
    markersize = 10
    win_colors = [male_color, male_color, female_color, female_color]
    lose_colors = [male_color, female_color, male_color, female_color]

    lose_beh_per_exp = []
    win_beh_per_exp = []
    for i in np.unique(trial_summary['exp_lose']):

        lose_beh_per_exp.append(trial_summary[beh_key_lose][(trial_summary['exp_lose'] == i) &
                                                        (trial_summary["draw"] == 0)].to_numpy())

        win_beh_per_exp.append(trial_summary[beh_key_win][(trial_summary['exp_lose'] == i) &
                                                        (trial_summary["draw"] == 0)].to_numpy())


    fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.15, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])

    ax.boxplot(lose_beh_per_exp, positions = np.unique(trial_summary['exp_lose'])-0.15, widths=0.2)
    ax.boxplot(win_beh_per_exp, positions = np.unique(trial_summary['exp_lose'])+0.15, widths=0.2)

    for enu, (win_sex, lose_sex) in enumerate(itertools.product(['m', 'f'], repeat=2)):
        lose_beh_count = trial_summary[beh_key_lose][(trial_summary["sex_win"] == win_sex) &
                                               (trial_summary["sex_lose"] == lose_sex) &
                                               (trial_summary["draw"] == 0)].to_numpy()
        win_beh_count = trial_summary[beh_key_win][(trial_summary["sex_win"] == win_sex) &
                                               (trial_summary["sex_lose"] == lose_sex) &
                                               (trial_summary["draw"] == 0)].to_numpy()

        lose_exp = trial_summary['exp_lose'][(trial_summary["sex_win"] == win_sex) &
                                         (trial_summary["sex_lose"] == lose_sex) &
                                         (trial_summary["draw"] == 0)].to_numpy()

        win_exp = trial_summary['exp_win'][(trial_summary["sex_win"] == win_sex) &
                                         (trial_summary["sex_lose"] == lose_sex) &
                                         (trial_summary["draw"] == 0)].to_numpy()

        ax.plot(lose_exp-0.15, lose_beh_count, 'o', color=lose_colors[enu], markeredgecolor=mek[enu],
                markersize=markersize, markeredgewidth=2)

        ax.plot(win_exp+0.15, win_beh_count, 'p', color=win_colors[enu], markeredgecolor=mek[enu],
                markersize=markersize, markeredgewidth=2)

    ax.set_xticks(np.unique(trial_summary['exp_lose']))
    ax.set_xticklabels(np.unique(trial_summary['exp_lose']))
    ax.set_xlabel('experience [trials]', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)
def main():
    beh_per_exp = []
    trial_summary = pd.read_csv('trial_summary.csv', index_col=0)
    chirp_notes = pd.read_csv('chirp_notes.csv', index_col=0)
    trial_summary = trial_summary[chirp_notes['good'] == 1]

    plot_rise_vs_chirp_count(trial_summary)

    plot_beh_count_per_pairing(trial_summary,
                               beh_key_win='chirps_win', beh_key_lose='chirps_lose',
                               ylabel='chirps [n]')
    plot_beh_count_per_pairing(trial_summary,
                               beh_key_win='rises_win', beh_key_lose='rises_lose',
                               ylabel='rises [n]')



    plot_beh_count_vs_meta(trial_summary,
                           beh_key_win='chirps_win', beh_key_lose='chirps_lose',
                           meta_key_win="size_win", meta_key_lose='size_lose',
                           xlabel=u'$\Delta$size [cm]')
    plot_beh_count_vs_meta(trial_summary,
                           beh_key_win='rises_win', beh_key_lose='rises_lose',
                           meta_key_win="size_win", meta_key_lose='size_lose',
                           xlabel=u'$\Delta$size [cm]')
    plot_beh_count_vs_meta(trial_summary,
                           beh_key_win='chirps_win', beh_key_lose='chirps_lose',
                           meta_key_win="EODf_win", meta_key_lose='EODf_lose',
                           xlabel=u'$\Delta$EODf [Hz]')
    plot_beh_count_vs_meta(trial_summary,
                           beh_key_win='rises_win', beh_key_lose='rises_lose',
                           meta_key_win="EODf_win", meta_key_lose='EODf_lose',
                           xlabel=u'$\Delta$EODf [Hz]')

    plot_beh_conut_vs_experience(trial_summary, beh_key_win='chirps_win', beh_key_lose='chirps_lose', ylabel='chirps [n]')
    plot_beh_conut_vs_experience(trial_summary, beh_key_win='rises_win', beh_key_lose='rises_lose', ylabel='rises [n]')

    plt.show()



if __name__ == '__main__':
    main()