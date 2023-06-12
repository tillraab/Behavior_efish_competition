import sys
import os
import scipy.stats as scp
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed

colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']
female_color, male_color = '#e74c3c', '#3498db'
Wc, Lc = 'darkgreen', '#3673A4'


def plot_rise_vs_chirp_count(trial_summary, trial_mask):
    fig = plt.figure(figsize=(20/2.54, 20/2.54))
    gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, height_ratios=[1, 3], width_ratios=[3, 1])
    ax = fig.add_subplot(gs[1, 0])

    ax.plot(trial_summary['rises_win'][(trial_summary["draw"] == 0) & trial_mask],
            trial_summary['chirps_win'][(trial_summary["draw"] == 0) & trial_mask], 'o', color=Wc, label='winner')
    ax.plot(trial_summary['rises_lose'][(trial_summary["draw"] == 0) & trial_mask],
            trial_summary['chirps_lose'][(trial_summary["draw"] == 0) & trial_mask], 'o', color=Lc, label='loster')
    ax.set_xlabel('rises [n]', fontsize=12)
    ax.set_ylabel('chirps [n]', fontsize=12)
    ax.tick_params(labelsize=10)

    ax_chirps = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_chirps.boxplot([trial_summary['chirps_win'][(trial_summary["draw"] == 0) & trial_mask],
                       trial_summary['chirps_lose'][(trial_summary["draw"] == 0) & trial_mask]], widths = .5, positions = [1, 2])
    ax_chirps.set_xticks([1, 2])
    ax_chirps.set_xticklabels(['Win', 'Lose'])
    plt.setp(ax_chirps.get_yticklabels(), visible=False)

    ax_rises = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_rises.boxplot([trial_summary['rises_win'][(trial_summary["draw"] == 0) & trial_mask],
                      trial_summary['rises_lose'][(trial_summary["draw"] == 0) & trial_mask]], widths = .5, positions = [1, 2], vert=False)
    ax_rises.set_yticks([1, 2])
    ax_rises.set_yticklabels(['Win', 'Lose'])
    plt.setp(ax_rises.get_xticklabels(), visible=False)

    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'rise_vs_chirp_count.png'), dpi=300)
    plt.close()


def plot_beh_count_per_pairing(trial_summary, trial_mask=None,
                               beh_key_win=None, beh_key_lose=None,
                               ylabel='y', save_str='random_plot_title'):

    mek = ['k', 'None', 'None', 'k']
    markersize = 12
    win_colors = [male_color, male_color, female_color, female_color]
    lose_colors = [male_color, female_color, male_color, female_color]

    if not hasattr(trial_mask, '__len__'):
        trial_mask = np.ones(len(trial_summary))
    win_count = []
    lose_count = []

    for win_sex, lose_sex in itertools.product(['m', 'f'], repeat=2):
        win_count.append(trial_summary[beh_key_win][(trial_summary["sex_win"] == win_sex) &
                                                    (trial_summary["sex_lose"] == lose_sex) &
                                                    (trial_summary["draw"] == 0) & trial_mask].to_numpy())
        lose_count.append(trial_summary[beh_key_lose][(trial_summary["sex_win"] == win_sex) &
                                                      (trial_summary["sex_lose"] == lose_sex) &
                                                      (trial_summary["draw"] == 0) & trial_mask].to_numpy())

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

    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', f'{save_str}.png'), dpi=300)
    plt.close()


def plot_meta_correlation(trial_summary, trial_mask, key1, key2, key1_name, key2_name, save_str='random_plot_title'):
    mek = ['k', 'None', 'None', 'k']
    markersize = 12
    win_colors = [male_color, male_color, female_color, female_color]
    lose_colors = [male_color, female_color, male_color, female_color]

    key1_collect = []
    key2_collect = []


    if 'chirp' in key1 or 'chirp' in key2:
        pass
    else:
        trial_mask = np.ones(len(trial_summary))

    for win_sex, lose_sex in itertools.product(['m', 'f'], repeat=2):
        k1 = trial_summary[key1][(trial_summary["sex_win"] == win_sex) &
                                 (trial_summary["sex_lose"] == lose_sex) &
                                 (trial_summary["draw"] == 0) & trial_mask].to_numpy()
        k2 = trial_summary[key2][(trial_summary["sex_win"] == win_sex) &
                                 (trial_summary["sex_lose"] == lose_sex) &
                                 (trial_summary["draw"] == 0) & trial_mask].to_numpy()
        mask = np.ones_like(k1, dtype=bool)
        mask[(k1 == -1) | (k2 == -1)] = 0
        k1 = k1[mask]
        k2 = k2[mask]
        key1_collect.append(k1)
        key2_collect.append(k2)

    fig = plt.figure(figsize=(20/2.54, 20/2.54))
    gs = gridspec.GridSpec(2, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0]))

    for i in range(len(key1_collect)):
        ax[0].plot(key1_collect[i], key2_collect[i], marker = 'p', color=win_colors[i], markeredgecolor=mek[i],
                markersize=markersize, markeredgewidth=2, linestyle='None')
        ax[1].plot(key1_collect[i], key2_collect[i], marker = 'o', color=lose_colors[i], markeredgecolor=mek[i],
                markersize=markersize, markeredgewidth=2, linestyle='None')

    ax[1].set_xlabel(f'{key1_name}', fontsize=12)
    ax[0].set_ylabel(f'{key2_name}', fontsize=12)
    ax[1].set_ylabel(f'{key2_name}', fontsize=12)

    plt.tick_params(labelsize=10)

    if True:
        r_coll = []
        p_coll = []
        print(f'\n{key1_name} - {key2_name}')
        for win_lose_key, sex in itertools.product(['sex_win', 'sex_lose'], ['m', 'f']):
            k1 = trial_summary[key1][(trial_summary[win_lose_key] == sex) & (trial_summary["draw"] == 0) & trial_mask].to_numpy()
            k2 = trial_summary[key2][(trial_summary[win_lose_key] == sex) & (trial_summary["draw"] == 0) & trial_mask].to_numpy()
            mask = np.ones_like(k1, dtype=bool)
            mask[np.isnan(k1) | np.isnan(k2)] = 0
            r, p = scp.spearmanr(k1[mask], k2[mask])
            r_coll.append(r)
            p_coll.append(p)
            print(f'{win_lose_key}: {sex} --> spearman-r={r:.2f} p={p:.3f}')
        k1 = trial_summary[key1][(trial_summary["draw"] == 0) & trial_mask].to_numpy()
        k2 = trial_summary[key2][(trial_summary["draw"] == 0) & trial_mask].to_numpy()
        mask = np.ones_like(k1, dtype=bool)
        mask[np.isnan(k1) | np.isnan(k2)] = 0
        r, p = scp.spearmanr(k1[mask], k2[mask])

        ax[0].text(1, 1, f'male win: spaerman-r = {r_coll[0]:.2f} p={p_coll[0]:.3f}\n'
                         f'female win: spaerman-r = {r_coll[1]:.2f} p={p_coll[1]:.3f}', ha='right', va='bottom', transform = ax[0].transAxes)
        ax[1].text(1, 1, f'male lose: spaerman-r = {r_coll[2]:.2f} p={p_coll[2]:.3f}\n'
                         f'female lose: spaerman-r = {r_coll[3]:.2f} p={p_coll[3]:.3f}', ha='right', va='bottom', transform = ax[1].transAxes)
        ax[1].text(1, -.1, f'all: spaerman-r = {r:.2f} p={p:.3f}', ha='right', va='top', transform = ax[1].transAxes)
        print(f'all --> spearman-r={r:.2f} p={p:.3f}')

    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', f'correlations_{key1}_{key2}.png'), dpi=300)
    plt.close()

def plot_beh_count_vs_dmeta(trial_summary, trial_mask=None,
                            beh_key_win=None, beh_key_lose=None,
                            meta_key_win=None, meta_key_lose=None,
                            xlabel='x', save_str='random_plot_title'):
    mek = ['k', 'None', 'None', 'k']
    markersize = 12
    win_colors = [male_color, male_color, female_color, female_color]
    lose_colors = [male_color, female_color, male_color, female_color]

    if not hasattr(trial_mask, '__len__'):
        trial_mask = np.ones(len(trial_summary))

    win_count = []
    lose_count = []

    win_meta = []
    lose_meta = []

    for win_sex, lose_sex in itertools.product(['m', 'f'], repeat=2):
        win_count.append(trial_summary[beh_key_win][(trial_summary["sex_win"] == win_sex) &
                                                    (trial_summary["sex_lose"] == lose_sex) &
                                                    (trial_summary["draw"] == 0) & trial_mask].to_numpy())
        lose_count.append(trial_summary[beh_key_lose][(trial_summary["sex_win"] == win_sex) &
                                                      (trial_summary["sex_lose"] == lose_sex) &
                                                      (trial_summary["draw"] == 0) & trial_mask].to_numpy())

        win_meta.append(trial_summary[meta_key_win][(trial_summary["sex_win"] == win_sex) &
                                                    (trial_summary["sex_lose"] == lose_sex) &
                                                    (trial_summary["draw"] == 0) & trial_mask].to_numpy())
        lose_meta.append(trial_summary[meta_key_lose][(trial_summary["sex_win"] == win_sex) &
                                                      (trial_summary["sex_lose"] == lose_sex) &
                                                      (trial_summary["draw"] == 0) & trial_mask].to_numpy())

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

    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', f'{save_str}.png'), dpi=300)
    plt.close()


def plot_beh_conut_vs_experience(trial_summary, trial_mask = None, beh_key_win='chirps_win', beh_key_lose='chirps_lose',
                                 ylabel='chirps [n]', save_str='random_plot_title'):
    mek = ['k', 'None', 'None', 'k']
    markersize = 10
    win_colors = [male_color, male_color, female_color, female_color]
    lose_colors = [male_color, female_color, male_color, female_color]

    if not hasattr(trial_mask, '__len__'):
        trial_mask = np.ones(len(trial_summary))

    lose_beh_per_exp = []
    win_beh_per_exp = []
    for i in np.unique(trial_summary['exp_lose']):

        lose_beh_per_exp.append(trial_summary[beh_key_lose][(trial_summary['exp_lose'] == i) &
                                                        (trial_summary["draw"] == 0) & trial_mask].to_numpy())

        win_beh_per_exp.append(trial_summary[beh_key_win][(trial_summary['exp_lose'] == i) &
                                                        (trial_summary["draw"] == 0) & trial_mask].to_numpy())


    fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.15, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])

    ax.boxplot(lose_beh_per_exp, positions = np.unique(trial_summary['exp_lose'])-0.15, widths=0.2)
    ax.boxplot(win_beh_per_exp, positions = np.unique(trial_summary['exp_lose'])+0.15, widths=0.2)

    for enu, (win_sex, lose_sex) in enumerate(itertools.product(['m', 'f'], repeat=2)):
        lose_beh_count = trial_summary[beh_key_lose][(trial_summary["sex_win"] == win_sex) &
                                               (trial_summary["sex_lose"] == lose_sex) &
                                               (trial_summary["draw"] == 0) & trial_mask].to_numpy()
        win_beh_count = trial_summary[beh_key_win][(trial_summary["sex_win"] == win_sex) &
                                               (trial_summary["sex_lose"] == lose_sex) &
                                               (trial_summary["draw"] == 0) & trial_mask].to_numpy()

        lose_exp = trial_summary['exp_lose'][(trial_summary["sex_win"] == win_sex) &
                                         (trial_summary["sex_lose"] == lose_sex) &
                                         (trial_summary["draw"] == 0) & trial_mask].to_numpy()

        win_exp = trial_summary['exp_win'][(trial_summary["sex_win"] == win_sex) &
                                         (trial_summary["sex_lose"] == lose_sex) &
                                         (trial_summary["draw"] == 0) & trial_mask].to_numpy()

        ax.plot(lose_exp-0.15, lose_beh_count, 'o', color=lose_colors[enu], markeredgecolor=mek[enu],
                markersize=markersize, markeredgewidth=2)

        ax.plot(win_exp+0.15, win_beh_count, 'p', color=win_colors[enu], markeredgecolor=mek[enu],
                markersize=markersize, markeredgewidth=2)

    ax.set_xticks(np.unique(trial_summary['exp_lose']))
    ax.set_xticklabels(np.unique(trial_summary['exp_lose']))
    ax.set_xlabel('experience [trials]', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)

    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', f'{save_str}.png'), dpi=300)
    plt.close()


def main(base_path):
    # ToDo: for chirp and rise analysis different datasets!!!
    # trial_summary = pd.read_csv(os.path.join(base_path, 'trial_summary.csv'), index_col=0)
    trial_summary = pd.read_csv(os.path.join(base_path, 'trial_summary.csv'), index_col=0)
    chirp_notes = pd.read_csv(os.path.join(base_path, 'chirp_notes.csv'), index_col=0)
    # trial_summary = trial_summary[chirp_notes['good'] == 1]
    trial_mask = chirp_notes['good'] == 1

    if True:
        print('')
        rc = np.concatenate((trial_summary['rises_win'][(trial_summary["draw"] == 0) & trial_mask],
                             trial_summary['rises_lose'][(trial_summary["draw"] == 0) & trial_mask]))
        cc = np.concatenate((trial_summary['chirps_win'][(trial_summary["draw"] == 0) & trial_mask],
                             trial_summary['chirps_lose'][(trial_summary["draw"] == 0) & trial_mask]))
        r, p = scp.spearmanr(rc, cc)
        print(f'Risescount - Chirpscount - all: Pearson-r={r:.2f} p={p:.3f}')

        r, p = scp.spearmanr(trial_summary['rises_win'][(trial_summary["draw"] == 0) & trial_mask],
                             trial_summary['chirps_win'][(trial_summary["draw"] == 0) & trial_mask])
        print(f'Risescount - Chirpscount - win: Pearson-r={r:.2f} p={p:.3f}')

        r, p = scp.spearmanr(trial_summary['rises_lose'][(trial_summary["draw"] == 0) & trial_mask],
                             trial_summary['chirps_lose'][(trial_summary["draw"] == 0) & trial_mask])
        print(f'Risescount - Chirpscount - lose: Pearson-r={r:.2f} p={p:.3f}')
    plot_rise_vs_chirp_count(trial_summary, trial_mask)

    if True:
        print('')
        chirps_lose_female_win = trial_summary['chirps_lose'][(trial_summary['sex_win'] == 'f') & (trial_summary["draw"] == 0) & trial_mask]
        chirps_lose_male_win = trial_summary['chirps_lose'][(trial_summary['sex_win'] == 'm') & (trial_summary["draw"] == 0) & trial_mask]

        U, p = scp.mannwhitneyu(chirps_lose_female_win, chirps_lose_male_win)
        print(f'Chirpscount - female win - male win: MW-U={U:.2f} p={p:.3f}')

        chirps_lose_female_lose = trial_summary['chirps_lose'][(trial_summary['sex_lose'] == 'f') & (trial_summary["draw"] == 0) & trial_mask]
        chirps_lose_male_lose = trial_summary['chirps_lose'][(trial_summary['sex_lose'] == 'm') & (trial_summary["draw"] == 0) & trial_mask]

        U, p = scp.mannwhitneyu(chirps_lose_female_lose, chirps_lose_male_lose)
        print(f'Chirpscount - female lose - male lose: MW-U={U:.2f} p={p:.3f}')
        ###################################################################################
        rises_lose_female_win = trial_summary['rises_lose'][(trial_summary['sex_win'] == 'f') & (trial_summary["draw"] == 0)]
        rises_lose_male_win = trial_summary['rises_lose'][(trial_summary['sex_win'] == 'm') & (trial_summary["draw"] == 0)]

        U, p = scp.mannwhitneyu(rises_lose_female_win, rises_lose_male_win)
        print(f'Risescount - female win - male win: MW-U={U:.2f} p={p:.3f}')

        rises_lose_female_lose = trial_summary['rises_lose'][(trial_summary['sex_lose'] == 'f') & (trial_summary["draw"] == 0)]
        rises_lose_male_lose = trial_summary['rises_lose'][(trial_summary['sex_lose'] == 'm') & (trial_summary["draw"] == 0)]

        U, p = scp.mannwhitneyu(rises_lose_female_lose, rises_lose_male_lose)
        print(f'Risescount - female lose - male lose: MW-U={U:.2f} p={p:.3f}')

    plot_beh_count_per_pairing(trial_summary, trial_mask,
                               beh_key_win='chirps_win', beh_key_lose='chirps_lose',
                               ylabel='chirps [n]', save_str='chirps_per_pairing')
    plot_beh_count_per_pairing(trial_summary, trial_mask=None,
                               beh_key_win='rises_win', beh_key_lose='rises_lose',
                               ylabel='rises [n]', save_str='rises_per_pairing')

    plot_beh_count_vs_dmeta(trial_summary, trial_mask,
                            beh_key_win='chirps_win', beh_key_lose='chirps_lose',
                            meta_key_win="size_win", meta_key_lose='size_lose',
                            xlabel=u'$\Delta$size [cm]', save_str='chirps_vs_dSize')
    plot_beh_count_vs_dmeta(trial_summary, trial_mask=None,
                            beh_key_win='rises_win', beh_key_lose='rises_lose',
                            meta_key_win="size_win", meta_key_lose='size_lose',
                            xlabel=u'$\Delta$size [cm]', save_str='rises_vs_dSize')
    plot_beh_count_vs_dmeta(trial_summary, trial_mask,
                            beh_key_win='chirps_win', beh_key_lose='chirps_lose',
                            meta_key_win="EODf_win", meta_key_lose='EODf_lose',
                            xlabel=u'$\Delta$EODf [Hz]', save_str='chirps_vs_dEODf')
    plot_beh_count_vs_dmeta(trial_summary, trial_mask=None,
                            beh_key_win='rises_win', beh_key_lose='rises_lose',
                            meta_key_win="EODf_win", meta_key_lose='EODf_lose',
                            xlabel=u'$\Delta$EODf [Hz]', save_str='rises_vs_dEODf')

    keys = ['dsize', 'dEODf', 'chirps_win', 'chirps_lose', 'rises_win', 'rises_lose', 'chase_count', 'contact_count', 'med_chase_dur', 'comp_dur0', 'comp_dur1']
    keys_names = [r'$\Delta$size$_{win}$', r'$\Delta$EODf$_{win}$', r'chirps$_{win}$', r'chirps$_{lose}$', 'rises$_{win}$', 'rises$_{lose}$', 'chase$_{n}$', 'contact$_{n}$', 'med_chase_dur', 'comp_dur0', 'comp_dur1']
    # for key1, key2 in itertools.combinations(keys, r = 2):
    for i, j in itertools.combinations(np.arange(len(keys)), r = 2):
        plot_meta_correlation(trial_summary, trial_mask, key1=keys[i], key2=keys[j],
                              key1_name=keys_names[i], key2_name=keys_names[j])

    # plot_meta_correlation(trial_summary, key1='med_chase_dur', key2='chirps_lose',
    #                       key1_name=r'chase duration$_{median}$ [s]', key2_name=r'chirps$_{lose}$')
    # plot_meta_correlation(trial_summary, key1='med_chase_dur', key2='rises_lose',
    #                       key1_name=r'chase duration$_{median}$ [s]', key2_name=r'rises$_{lose}$')

    # if True:
    #     ### chirp count vs. dSize ###
    #     for key in ['chirps_lose', 'chirps_win', 'rises_win', 'rises_lose']:
    #         print('')
    #         lose_chirps_male_win = trial_summary[key][(trial_summary['sex_win'] == 'm') & (trial_summary["draw"] == 0)]
    #         lose_size_male_win = trial_summary['size_lose'][(trial_summary['sex_win'] == 'm') & (trial_summary["draw"] == 0)]
    #         win_size_male_win = trial_summary['size_win'][(trial_summary['sex_win'] == 'm') & (trial_summary["draw"] == 0)]
    #
    #         r, p = scp.pearsonr((lose_size_male_win - win_size_male_win)*-1, lose_chirps_male_win)
    #         print(f'(Male win) {key} - dSize: Pearson-r={r:.2f} p={p:.3f}')
    #
    #         lose_chirps_female_win = trial_summary[key][(trial_summary['sex_win'] == 'f') & (trial_summary["draw"] == 0)]
    #         lose_size_female_win = trial_summary['size_lose'][(trial_summary['sex_win'] == 'f') & (trial_summary["draw"] == 0)]
    #         win_size_female_win = trial_summary['size_win'][(trial_summary['sex_win'] == 'f') & (trial_summary["draw"] == 0)]
    #
    #         r, p = scp.pearsonr(lose_chirps_female_win, lose_size_female_win - win_size_female_win)
    #         print(f'(Female win) {key} - dSize: Pearson-r={r:.2f} p={p:.3f}')
    #
    #         lose_chirps_male_lose = trial_summary[key][(trial_summary['sex_lose'] == 'm') & (trial_summary["draw"] == 0)]
    #         lose_size_male_lose = trial_summary['size_lose'][(trial_summary['sex_lose'] == 'm') & (trial_summary["draw"] == 0)]
    #         win_size_male_lose = trial_summary['size_win'][(trial_summary['sex_lose'] == 'm') & (trial_summary["draw"] == 0)]
    #
    #         r, p = scp.pearsonr(lose_chirps_male_lose, lose_size_male_lose - win_size_male_lose)
    #         print(f'(Male lose) {key} - dSize: Pearson-r={r:.2f} p={p:.3f}')
    #
    #         lose_chirps_female_lose = trial_summary[key][(trial_summary['sex_lose'] == 'f') & (trial_summary["draw"] == 0)]
    #         lose_size_female_lose = trial_summary['size_lose'][(trial_summary['sex_lose'] == 'f') & (trial_summary["draw"] == 0)]
    #         win_size_female_lose = trial_summary['size_win'][(trial_summary['sex_lose'] == 'f') & (trial_summary["draw"] == 0)]
    #
    #         r, p = scp.pearsonr(lose_chirps_female_lose, lose_size_female_lose - win_size_female_lose)
    #         print(f'(Female lose) {key} - dSize: Pearson-r={r:.2f} p={p:.3f}')
    #
    #         all_lose_chrips = trial_summary[key][(trial_summary["draw"] == 0)]
    #         all_lose_size = trial_summary['size_lose'][(trial_summary["draw"] == 0)]
    #         all_win_size = trial_summary['size_win'][(trial_summary["draw"] == 0)]
    #         r, p = scp.pearsonr(all_lose_chrips, all_lose_size - all_win_size)
    #
    #         print(f'(all) {key} - dSize: Pearson-r={r:.2f} p={p:.3f}')

    plot_beh_conut_vs_experience(trial_summary, trial_mask, beh_key_win='chirps_win', beh_key_lose='chirps_lose',
                                 ylabel='chirps [n]', save_str='chirps_by_experince')
    plot_beh_conut_vs_experience(trial_summary, trial_mask=None, beh_key_win='rises_win', beh_key_lose='rises_lose', ylabel='rises [n]',
                                 save_str='rises_by_experince')

    if True:
        for key in ['chirps_lose', 'chirps_win', 'rises_lose', 'rises_win']:
            print('')
            if 'chirps' in key:
                lose_events = trial_summary[key][(trial_summary["draw"] == 0) & trial_mask]
                lose_exp = trial_summary['exp_lose'][(trial_summary["draw"] == 0) & trial_mask]
                win_exp = trial_summary['exp_win'][(trial_summary["draw"] == 0) & trial_mask]
            else:
                lose_events = trial_summary[key][(trial_summary["draw"] == 0)]
                lose_exp = trial_summary['exp_lose'][(trial_summary["draw"] == 0)]
                win_exp = trial_summary['exp_win'][(trial_summary["draw"] == 0)]

            r, p = scp.pearsonr(lose_events, lose_exp)
            print(f'(all) {key} - lose exp: Pearson-r={r:.2f} p={p:.3f}')
            r, p = scp.pearsonr(lose_events, win_exp)
            print(f'(all) {key} - win exp: Pearson-r={r:.2f} p={p:.3f}')

    plt.show()



if __name__ == '__main__':
    main(sys.argv[1])