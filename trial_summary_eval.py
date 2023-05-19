import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed

colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']
female_color, male_color = '#e74c3c', '#3498db'
Wc, Lc = 'darkgreen', '#3673A4'


def plot_chirp_rise_count_per_pairing(trial_summary):
    win_chirps = []
    lose_chirps = []

    win_rises = []
    lose_rises = []

    for win_sex, lose_sex in itertools.product(['m', 'f'], repeat=2):
        win_chirps.append(trial_summary['chirps_win'][(trial_summary["sex_win"] == win_sex) &
                                                      (trial_summary["sex_lose"] == lose_sex) &
                                                      (trial_summary["draw"] == 0)].to_numpy())
        lose_chirps.append(trial_summary['chirps_lose'][(trial_summary["sex_win"] == win_sex) &
                                                        (trial_summary["sex_lose"] == lose_sex) &
                                                        (trial_summary["draw"] == 0)].to_numpy())
        win_rises.append(trial_summary['rises_win'][(trial_summary["sex_win"] == win_sex) &
                                                      (trial_summary["sex_lose"] == lose_sex) &
                                                      (trial_summary["draw"] == 0)].to_numpy())
        lose_rises.append(trial_summary['rise_lose'][(trial_summary["sex_win"] == win_sex) &
                                                        (trial_summary["sex_lose"] == lose_sex) &
                                                        (trial_summary["draw"] == 0)].to_numpy())


    fig = plt.figure(figsize=(20/2.54, 12/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
    ax = fig.add_subplot(gs[0, 0])
    ax.boxplot(win_chirps, positions=np.arange(len(win_chirps))-0.15, widths= .2, sym='')
    ax.boxplot(lose_chirps, positions=np.arange(len(lose_chirps))+0.15, widths= .2, sym='')

    ax.set_xticks(np.arange(len(win_chirps)))
    # ax.set_xticklabels([u'\u2642\u2642', u'\u2642\u2640', u'\u2640\u2642', u'\u2640\u2640'])
    ax.set_xticklabels(['mm', 'mf', 'fm', 'ff'])
    y0, y1 = ax.get_ylim()
    for i in range(len(win_chirps)):
        ax.text(i, y1, f'n={len(win_chirps[i]):.0f}', fontsize=10, ha='center', va='bottom')
    ax.set_ylim(top = y1*1.1)
    ax.set_ylabel('chirps [n]', fontsize=12)
    plt.tick_params(labelsize=10)

    fig = plt.figure(figsize=(20/2.54, 12/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
    ax = fig.add_subplot(gs[0, 0])
    ax.boxplot(win_rises, positions=np.arange(len(win_rises))-0.15, widths= .2, sym='')
    ax.boxplot(lose_rises, positions=np.arange(len(lose_rises))+0.15, widths= .2, sym='')

    ax.set_xticks(np.arange(len(win_rises)))
    # ax.set_xticklabels([u'\u2642\u2642', u'\u2642\u2640', u'\u2640\u2642', u'\u2640\u2640'])
    ax.set_xticklabels(['mm', 'mf', 'fm', 'ff'])
    y0, y1 = ax.get_ylim()
    for i in range(len(win_rises)):
        ax.text(i, y1, f'n={len(win_rises[i]):.0f}', fontsize=10, ha='center', va='bottom')
    ax.set_ylim(top = y1*1.1)
    ax.set_ylabel('rises [n]', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.show()

def plot_chirp_rise_count_per_vs_size_diff(trial_summary):
    win_chirps = []
    lose_chirps = []

    win_rises = []
    lose_rises = []

    d_size = []

    for win_sex, lose_sex in itertools.product(['m', 'f'], repeat=2):
        win_chirps.append(trial_summary['chirps_win'][(trial_summary["sex_win"] == win_sex) &
                                                      (trial_summary["sex_lose"] == lose_sex) &
                                                      (trial_summary["draw"] == 0)].to_numpy())
        lose_chirps.append(trial_summary['chirps_lose'][(trial_summary["sex_win"] == win_sex) &
                                                        (trial_summary["sex_lose"] == lose_sex) &
                                                        (trial_summary["draw"] == 0)].to_numpy())
        win_rises.append(trial_summary['rises_win'][(trial_summary["sex_win"] == win_sex) &
                                                      (trial_summary["sex_lose"] == lose_sex) &
                                                      (trial_summary["draw"] == 0)].to_numpy())
        lose_rises.append(trial_summary['rise_lose'][(trial_summary["sex_win"] == win_sex) &
                                                        (trial_summary["sex_lose"] == lose_sex) &
                                                        (trial_summary["draw"] == 0)].to_numpy())

        w_size = trial_summary['size_win'][(trial_summary["sex_win"] == win_sex) &
                                           (trial_summary["sex_lose"] == lose_sex) &
                                           (trial_summary["draw"] == 0)].to_numpy()
        l_size = trial_summary['size_lose'][(trial_summary["sex_win"] == win_sex) &
                                            (trial_summary["sex_lose"] == lose_sex) &
                                            (trial_summary["draw"] == 0)].to_numpy()

        d_size.append(w_size-l_size)
    embed()
    quit()

    fig = plt.figure(figsize=(20/2.54, 12/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
    ax = fig.add_subplot(gs[0, 0])
    mek = ['k', 'None', 'None', 'k']

    c = [male_color, male_color, female_color, female_color]
    for i in range(len(lose_rises)):
        ax.plot(d_size[i]*-1, lose_rises[i], 'p', color=c[i], markeredgecolor=mek[i], markersize=8)

    ax.set_ylabel('rises [n]', fontsize=12)
    plt.tick_params(labelsize=10)


def main():
    trial_summary = pd.read_csv('trial_summary.csv', index_col=0)

    plot_chirp_rise_count_per_pairing(trial_summary)

    plot_chirp_rise_count_per_vs_size_diff(trial_summary)


    pass

if __name__ == '__main__':
    main()