import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import pandas as pd
import scipy.stats as scp
from IPython import embed
from event_time_correlations import load_and_converete_boris_events, kde, gauss

female_color, male_color = '#e74c3c', '#3498db'

def iei_analysis(event_times, win_sex, lose_sex, kernal_w, title=''):
    # ToDo: finish this !!!
    iei = []
    weighted_mean_iei = []
    median_iei = []
    for i in range(len(event_times)):
        trial_iei = np.diff(event_times[i][event_times[i] <= 3600*3])
        iei.append(trial_iei)

        if len(trial_iei) == 0:
            weighted_mean_iei.append(np.nan)
            median_iei.append(np.nan)
        else:
            weighted_mean_iei.append(np.sum((trial_iei) * trial_iei) / np.sum(trial_iei))
            median_iei.append(np.median(trial_iei))

    weighted_mean_iei = np.array(weighted_mean_iei)
    median_iei = np.array(median_iei)

    fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
    gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.95, top=0.9)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1], sharey=ax[0], sharex=ax[0]))
    ax.append(fig.add_subplot(gs[1, 0], sharey=ax[0], sharex=ax[0]))
    ax.append(fig.add_subplot(gs[1, 1], sharey=ax[0], sharex=ax[0]))

    for i in range(len(iei)):
        if win_sex[i] == 'm':
            if lose_sex[i] == 'm':
                color, linestyle = male_color, '-'
                sp = 0
            else:
                color, linestyle = male_color, '--'
                sp = 1
        else:
            if lose_sex[i] == 'm':
                color, linestyle = female_color, '--'
                sp = 2
            else:
                color, linestyle = female_color, '-'
                sp = 3

        conv_y = np.arange(0, np.percentile(np.hstack(iei), 80), .5)
        kde_array = kde(iei[i], conv_y, kernal_w=kernal_w, kernal_h=1)

        # kde_array /= np.sum(kde_array)
        ax[sp].plot(conv_y, kde_array, zorder=2, color=color, linestyle=linestyle, lw=2)

    # ax_m = ax[0].twinx()
    # ax_m.boxplot([weighted_mean_iei[(win_sex == 'm') & (win_sex == 'm') & ~np.isnan(weighted_mean_iei)],
    #               median_iei[(win_sex == 'm') & (win_sex == 'm') & ~np.isnan(median_iei)]], sym='', vert=False)

    ax[0].set_xlim(conv_y[0], conv_y[-1])
    ax[0].set_ylabel('KDE', fontsize=12)
    ax[2].set_ylabel('KDE', fontsize=12)
    ax[2].set_xlabel('time [s]', fontsize=12)
    ax[3].set_xlabel('time [s]', fontsize=12)
    fig.suptitle(title, fontsize=12)

    for a in ax:
        a.tick_params(labelsize=10)

    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.setp(ax[3].get_yticklabels(), visible=False)

    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[1].get_xticklabels(), visible=False)

    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'event_meta', f'{title}_iei.png'), dpi=300)
    plt.close()
    # plt.show()

    # for iei, kernal_w in zip([ici_lose, ici_win, iri_lose, iri_win],
    #                          [1, 1, 5, 50]):
    #
    #     fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
    #     gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.95, top=0.95)
    #     ax = []
    #     ax.append(fig.add_subplot(gs[0, 0]))
    #     ax.append(fig.add_subplot(gs[0, 1], sharey=ax[0], sharex=ax[0]))
    #     ax.append(fig.add_subplot(gs[1, 0], sharey=ax[0], sharex=ax[0]))
    #     ax.append(fig.add_subplot(gs[1, 1], sharey=ax[0], sharex=ax[0]))
    #
    #     for i in range(len(iei)):
    #         if win_sex[i] == 'm':
    #             if lose_sex[i] == 'm':
    #                 color, linestyle = male_color, '-'
    #                 sp = 0
    #             else:
    #                 color, linestyle = male_color, '--'
    #                 sp = 1
    #         else:
    #             if lose_sex[i] == 'm':
    #                 color, linestyle = female_color, '--'
    #                 sp = 2
    #             else:
    #                 color, linestyle = female_color, '-'
    #                 sp = 3
    #
    #
    #         conv_y = np.arange(0, np.percentile(np.hstack(iei), 90), .5)
    #         kde_array = kde(iei[i], conv_y, kernal_w=kernal_w, kernal_h=1)
    #
    #         # kde_array /= np.sum(kde_array)
    #         ax[sp].plot(conv_y, kde_array, zorder=2, color=color, linestyle=linestyle, lw=2)
    #
    #     plt.setp(ax[1].get_yticklabels(), visible=False)
    #     plt.setp(ax[3].get_yticklabels(), visible=False)
    #
    #
    #     plt.setp(ax[0].get_xticklabels(), visible=False)
    #     plt.setp(ax[1].get_xticklabels(), visible=False)
    #     plt.show()


def relative_rate_progression(all_event_t, title=''):
    stop_t = 3*60*60
    snippet_len = 15*60

    snippet_starts = np.arange(0, stop_t, snippet_len)
    all_snippet_ratio = []
    for event_t in all_event_t:
        if len(event_t) == 0:
            continue
        expected_snippet_count = len(event_t[event_t <= stop_t]) / (stop_t / snippet_len)

        snippet_ratio = []
        for s0 in snippet_starts:
            snippet_count = len(event_t[(event_t >= s0) & (event_t < s0 + snippet_len)])
            snippet_ratio.append(snippet_count/expected_snippet_count)

        all_snippet_ratio.append(snippet_ratio)
    all_snippet_ratio = np.array(all_snippet_ratio)

    fig = plt.figure(figsize=(20/2.54, 12/2.54))
    gs = gridspec.GridSpec(1, 1, left=.1, bottom=.1, right=0.95, top=0.95)
    ax = fig.add_subplot(gs[0, 0])

    plot_t = np.repeat(snippet_starts, 2)
    plot_t[1::2] += snippet_len

    for event_ratios in all_snippet_ratio:
        plot_ratios = np.repeat(event_ratios, 2)
        ax.plot(plot_t / 3600, plot_ratios, color='grey', lw=1, alpha=0.5)
        # ax.plot(snippet_starts + snippet_len/2, event_ratios)
    mean_ratio = np.median(all_snippet_ratio, axis=0)
    plot_mean_ratio = np.repeat(mean_ratio, 2)
    ax.plot(plot_t / 3600, plot_mean_ratio, color='k', lw=3)
    ax.plot(plot_t / 3600, np.ones_like(plot_t), linestyle='dotted', lw=2, color='k')

    ax.set_xlabel('time [h]', fontsize=12)
    ax.set_ylabel('norm. event rate', fontsize=12)
    ax.set_title(title)
    ax.tick_params(labelsize=10)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 5)

    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'event_meta', f'{title}_progression.png'), dpi=300)
    plt.close()
    # plt.show()

    x = np.hstack(all_snippet_ratio)
    y = np.hstack(np.tile(snippet_starts, (all_snippet_ratio.shape[0], 1)))

    r, p = scp.pearsonr(x, y)

    print(f'Progression {title}: pearson-r={r:.2f} p={p:.3f}')





def main(base_path):
    if not os.path.exists(os.path.join(os.path.split(__file__)[0], 'figures', 'event_meta')):
        os.makedirs(os.path.join(os.path.split(__file__)[0], 'figures', 'event_meta'))

    if not os.path.exists(os.path.join(os.path.split(__file__)[0], 'figures', 'event_time_corr')):
        os.makedirs(os.path.join(os.path.split(__file__)[0], 'figures', 'event_time_corr'))


    trial_summary = pd.read_csv(os.path.join(base_path, 'trial_summary.csv'), index_col=0)
    chirp_notes = pd.read_csv(os.path.join(base_path, 'chirp_notes.csv'), index_col=0)
    trial_mask = chirp_notes['good'] == 1
    # trial_summary = trial_summary[chirp_notes['good'] == 1]

    all_rise_times_lose = []
    all_rise_times_win = []
    all_chirp_times_lose = []
    all_chirp_times_win = []
    win_sex = []
    lose_sex = []

    all_contact_t = []
    all_ag_on_t = []
    all_ag_off_t = []

    for index, trial in trial_summary.iterrows():
        print(index, len(trial_summary))
        got_boris = False

        trial_path = os.path.join(base_path, trial['recording'])

        if trial['group'] < 3:
            continue
        if trial['draw'] == 1:
            continue
        if os.path.exists(os.path.join(trial_path, 'led_idxs.csv')):
            got_boris = True
        if os.path.exists(os.path.join(trial_path, 'LED_frames.npy')):
            got_boris = True

        ids = np.load(os.path.join(trial_path, 'analysis', 'ids.npy'))
        times = np.load(os.path.join(trial_path, 'times.npy'))
        sorter = -1 if trial['win_ID'] != ids[0] else 1

        ### event times --> BORIS behavior
        if got_boris:
            contact_t_GRID, ag_on_off_t_GRID, led_idx, led_frames = \
                load_and_converete_boris_events(trial_path, trial['recording'], sr=20_000)
            all_contact_t.append(contact_t_GRID)
            all_ag_on_t.append(ag_on_off_t_GRID[:, 0])
            all_ag_off_t.append(ag_on_off_t_GRID[:, 1])
        else:
            all_contact_t.append(np.array([]))
            all_ag_on_t.append(np.array([]))
            all_ag_off_t.append(np.array([]))

        ### communication
        if not os.path.exists(os.path.join(trial_path, 'chirp_times_cnn.npy')):
            continue
        chirp_t = np.load(os.path.join(trial_path, 'chirp_times_cnn.npy'))
        chirp_ids = np.load(os.path.join(trial_path, 'chirp_ids_cnn.npy'))
        chirp_times = [chirp_t[chirp_ids == trial['win_ID']], chirp_t[chirp_ids == trial['lose_ID']]]


        rise_idx = np.load(os.path.join(trial_path, 'analysis', 'rise_idx.npy'))[::sorter]
        rise_idx_int = [np.array(rise_idx[i][~np.isnan(rise_idx[i])], dtype=int) for i in range(len(rise_idx))]
        rise_times = [times[rise_idx_int[0]], times[rise_idx_int[1]]]

        all_rise_times_lose.append(rise_times[1])
        all_rise_times_win.append(rise_times[0])

        if trial_mask[index]:
            all_chirp_times_lose.append(chirp_times[1])
            all_chirp_times_win.append(chirp_times[0])
        else:
            all_chirp_times_lose.append(np.array([]))
            all_chirp_times_win.append(np.array([]))

        win_sex.append(trial['sex_win'])
        lose_sex.append(trial['sex_lose'])

    win_sex = np.array(win_sex)
    lose_sex = np.array(lose_sex)

    iei_analysis(all_chirp_times_lose, win_sex, lose_sex, kernal_w=1, title=r'chirps$_{lose}$')
    iei_analysis(all_chirp_times_win,  win_sex, lose_sex, kernal_w=1, title=r'chirps$_{win}$')
    iei_analysis(all_rise_times_lose, win_sex, lose_sex, kernal_w=5, title=r'rises$_{lose}$')
    iei_analysis(all_rise_times_win, win_sex, lose_sex, kernal_w=50, title=r'rises$_{win}$')

    print('')
    relative_rate_progression(all_chirp_times_lose, title=r'chirp$_{lose}$')
    relative_rate_progression(all_chirp_times_win, title=r'chirp$_{win}$')
    relative_rate_progression(all_rise_times_lose, title=r'rises$_{lose}$')
    relative_rate_progression(all_rise_times_win, title=r'rises$_{win}$')

    relative_rate_progression(all_contact_t, title=r'contact')
    relative_rate_progression(all_ag_on_t, title=r'chasing')


    #############################################################################
    for all_event_t, event_name in zip([all_chirp_times_lose, all_chirp_times_win, all_rise_times_lose, all_rise_times_win],
                                       [r'chirps$_{lose}$', r'chirps$_{win}$', r'rises$_{lose}$', r'rises$_{win}$']):
        print('')
        all_pre_chase_event_mask = []
        all_chase_event_mask = []
        all_end_chase_event_mask = []
        all_after_chase_event_mask = []
        all_before_contact_event_mask = []
        all_after_contact_event_mask = []

        all_pre_chase_time = []
        all_chase_time = []
        all_end_chase_time = []
        all_after_chase_time = []
        all_before_contact_time = []
        all_after_contact_time = []

        video_trial_win_sex = []
        video_trial_lose_sex = []

        time_tol = 5

        for enu, contact_t, ag_on_t, ag_off_t, event_times in zip(
                np.arange(len(all_contact_t)), all_contact_t, all_ag_on_t, all_ag_off_t, all_event_t):

            if len(ag_on_t) == 0:
                continue

            if len(event_times) == 0:
                continue

            pre_chase_event_mask = np.zeros_like(event_times)
            chase_event_mask = np.zeros_like(event_times)
            end_chase_event_mask = np.zeros_like(event_times)
            after_chase_event_mask = np.zeros_like(event_times)

            video_trial_win_sex.append(win_sex[enu])
            video_trial_lose_sex.append(lose_sex[enu])

            for chase_on_t, chase_off_t in zip(ag_on_t, ag_off_t):
                pre_chase_event_mask[(event_times >= chase_on_t - time_tol) & (event_times < chase_on_t)] = 1
                chase_event_mask[(event_times >= chase_on_t) & (event_times < chase_off_t - time_tol)] = 1
                end_chase_event_mask[(event_times >= chase_off_t - time_tol) & (event_times < chase_off_t)] = 1
                after_chase_event_mask[(event_times >= chase_off_t) & (event_times < chase_off_t + time_tol)] = 1

            all_pre_chase_event_mask.append(pre_chase_event_mask)
            all_chase_event_mask.append(chase_event_mask)
            all_end_chase_event_mask.append(end_chase_event_mask)
            all_after_chase_event_mask.append(after_chase_event_mask)

            all_pre_chase_time.append(len(ag_on_t) * time_tol)
            chasing_dur = (ag_off_t - ag_on_t) - time_tol
            chasing_dur[chasing_dur < 0 ] = 0
            all_chase_time.append(np.sum(chasing_dur))
            all_end_chase_time.append(len(ag_on_t) * time_tol)
            all_after_chase_time.append(len(ag_on_t) * time_tol)

            before_countact_event_mask = np.zeros_like(event_times)
            after_countact_event_mask = np.zeros_like(event_times)
            for ct in contact_t:
                before_countact_event_mask[(event_times >= ct-time_tol) & (event_times < ct)] = 1
                after_countact_event_mask[(event_times >= ct) & (event_times < ct+time_tol)] = 1
            all_before_contact_event_mask.append(before_countact_event_mask)
            all_after_contact_event_mask.append(after_countact_event_mask)

            all_before_contact_time.append(len(contact_t) * time_tol)
            all_after_contact_time.append(len(contact_t) * time_tol)

        all_pre_chase_time = np.array(all_pre_chase_time)
        all_chase_time = np.array(all_chase_time)
        all_end_chase_time = np.array(all_end_chase_time)
        all_after_chase_time = np.array(all_after_chase_time)
        all_before_contact_time = np.array(all_before_contact_time)
        all_after_contact_time = np.array(all_after_contact_time)

        video_trial_win_sex = np.array(video_trial_win_sex)
        video_trial_lose_sex = np.array(video_trial_lose_sex)

        all_pre_chase_time_ratio = all_pre_chase_time / (3*60*60)
        all_chase_time_ratio = all_chase_time / (3*60*60)
        all_end_chase_time_ratio = all_end_chase_time / (3*60*60)
        all_after_chase_time_ratio = all_after_chase_time / (3*60*60)
        all_before_countact_time_ratio = all_before_contact_time / (3*60*60)
        all_after_countact_time_ratio = all_after_contact_time / (3*60*60)

        all_pre_chase_event_ratio = np.array(list(map(lambda x: np.sum(x)/len(x), all_pre_chase_event_mask)))
        all_chase_event_ratio = np.array(list(map(lambda x: np.sum(x)/len(x), all_chase_event_mask)))
        all_end_chase_event_ratio = np.array(list(map(lambda x: np.sum(x)/len(x), all_end_chase_event_mask)))
        all_after_chase_event_ratio = np.array(list(map(lambda x: np.sum(x)/len(x), all_after_chase_event_mask)))
        all_before_countact_event_ratio = np.array(list(map(lambda x: np.sum(x)/len(x), all_before_contact_event_mask)))
        all_after_countact_event_ratio = np.array(list(map(lambda x: np.sum(x)/len(x), all_after_contact_event_mask)))

        for x, y, name in [[all_pre_chase_event_ratio, all_pre_chase_time_ratio, 'pre chase'],
                           [all_chase_event_ratio, all_chase_time_ratio, 'while chase'],
                           [all_end_chase_event_ratio, all_end_chase_time_ratio, 'end chase'],
                           [all_after_chase_event_ratio, all_after_chase_time_ratio, 'after chase'],
                           [all_before_countact_event_ratio, all_before_countact_time_ratio, 'pre contact'],
                           [all_after_countact_event_ratio, all_after_countact_time_ratio, 'post contact']]:
            t, p = scp.ttest_rel(x, y)
            print(f'{event_name} {name}: t={t:.2f} p={p:.3f}')


        fig = plt.figure(figsize=(20/2.54, 12/2.54))
        gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.15, right=0.95, top=0.9)
        ax = fig.add_subplot(gs[0, 0])
        ax_pie = fig.add_subplot(gs[0, 1])

        ax.boxplot([all_pre_chase_event_ratio/all_pre_chase_time_ratio,
                    all_chase_event_ratio/all_chase_time_ratio,
                    all_end_chase_event_ratio/all_end_chase_time_ratio,
                    all_after_chase_event_ratio/all_after_chase_time_ratio,
                    all_before_countact_event_ratio/all_before_countact_time_ratio,
                    all_after_countact_event_ratio/all_after_countact_time_ratio], positions=np.arange(6), sym='', zorder=2)

        ylim = list(ax.get_ylim())
        ylim[0] = -.1 if ylim[0] < -.1 else ylim[0]
        ylim[1] = 1.1 if ylim[1] < 1.1 else ylim[1]
        ##############################################################################
        for sex_w, sex_l in itertools.product(['m', 'f'], repeat=2):
            mec = 'k' if sex_w == sex_l else 'None'
            if 'lose' in event_name:
                marker='o'
                c = male_color if sex_l == 'm' else female_color
            elif "win" in event_name:
                marker='p'
                c = male_color if sex_w == 'm' else female_color
            else:
                print('error')
                embed()
                quit()
            values = np.array(all_pre_chase_event_ratio/all_pre_chase_time_ratio)[(video_trial_win_sex == sex_w) & (video_trial_lose_sex == sex_l)]
            ax.plot(np.ones_like(values) * 0, values, marker=marker, linestyle='None', color=c, mec=mec, markersize=8, zorder=1)

            values = np.array(all_chase_event_ratio/all_chase_time_ratio)[(video_trial_win_sex == sex_w) & (video_trial_lose_sex == sex_l)]
            ax.plot(np.ones_like(values) * 1, values, marker=marker, linestyle='None', color=c, mec=mec, markersize=8, zorder=1)

            values = np.array(all_end_chase_event_ratio/all_end_chase_time_ratio)[(video_trial_win_sex == sex_w) & (video_trial_lose_sex == sex_l)]
            ax.plot(np.ones_like(values) * 2, values, marker=marker, linestyle='None', color=c, mec=mec, markersize=8, zorder=1)

            values = np.array(all_after_chase_event_ratio/all_after_chase_time_ratio)[(video_trial_win_sex == sex_w) & (video_trial_lose_sex == sex_l)]
            ax.plot(np.ones_like(values) * 3, values, marker=marker, linestyle='None', color=c, mec=mec, markersize=8, zorder=1)

            values = np.array(all_before_countact_event_ratio/all_before_countact_time_ratio)[(video_trial_win_sex == sex_w) & (video_trial_lose_sex == sex_l)]
            ax.plot(np.ones_like(values) * 4, values, marker=marker, linestyle='None', color=c, mec=mec, markersize=8, zorder=1)

            values = np.array(all_after_countact_event_ratio/all_after_countact_time_ratio)[(video_trial_win_sex == sex_w) & (video_trial_lose_sex == sex_l)]
            ax.plot(np.ones_like(values) * 5, values, marker=marker, linestyle='None', color=c, mec=mec, markersize=8, zorder=1)

        # ax.plot(np.ones_like(all_pre_chase_event_ratio) * 0, all_pre_chase_event_ratio/all_pre_chase_time_ratio, 'ok')
        # ax.plot(np.ones_like(all_chase_event_ratio) * 1, all_chase_event_ratio/all_chase_time_ratio, 'ok')
        # ax.plot(np.ones_like(all_end_chase_event_ratio) * 2, all_end_chase_event_ratio/all_end_chase_time_ratio, 'ok')
        # ax.plot(np.ones_like(all_after_chase_event_ratio) * 3, all_after_chase_event_ratio/all_after_chase_time_ratio, 'ok')

        # ax.plot(np.ones_like(all_before_countact_event_ratio) * 4, all_before_countact_event_ratio/all_before_countact_time_ratio, 'ok')
        ##############################################################################

        ax.plot(np.arange(7)-1, np.ones(7), linestyle='dotted', lw=2, color='k')
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(ylim[0], ylim[1])

        ax.set_ylabel(r'rel. count$_{event}$ / rel. time$_{event}$', fontsize=12)
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels([r'chase$_{before}$', r'chasing', r'chase$_{end}$', r'chase$_{after}$', 'contact$_{before}$', 'contact$_{after}$'], rotation=45)
        ax.tick_params(labelsize=10)
        # ax.set_title(event_name)
        fig.suptitle(f'{event_name}: n={len(np.hstack(all_event_t))}')
        # plt.show()

        ###############################################
        flat_pre_chase_event_mask = np.hstack(all_pre_chase_event_mask)
        flat_chase_event_mask = np.hstack(all_chase_event_mask)
        flat_end_chase_event_mask = np.hstack(all_end_chase_event_mask)
        flat_after_chase_event_mask = np.hstack(all_after_chase_event_mask)
        flat_before_countact_event_mask = np.hstack(all_before_contact_event_mask)
        flat_after_countact_event_mask = np.hstack(all_after_contact_event_mask)

        flat_pre_chase_event_mask[(flat_before_countact_event_mask == 1) | (flat_after_countact_event_mask == 1)] = 0
        flat_chase_event_mask[(flat_before_countact_event_mask == 1) | (flat_after_countact_event_mask == 1)] = 0
        flat_end_chase_event_mask[(flat_before_countact_event_mask == 1) | (flat_after_countact_event_mask == 1)] = 0
        flat_after_chase_event_mask[(flat_before_countact_event_mask == 1) | (flat_after_countact_event_mask == 1)] = 0

        event_context_values = [np.sum(flat_pre_chase_event_mask) / len(flat_pre_chase_event_mask),
                                np.sum(flat_chase_event_mask) / len(flat_chase_event_mask),
                                np.sum(flat_end_chase_event_mask) / len(flat_end_chase_event_mask),
                                np.sum(flat_after_chase_event_mask) / len(flat_after_chase_event_mask),
                                np.sum(flat_before_countact_event_mask) / len(flat_before_countact_event_mask),
                                np.sum(flat_after_countact_event_mask) / len(flat_after_countact_event_mask)]

        event_context_values.append(1 - np.sum(event_context_values))

        time_context_values = [np.sum(all_pre_chase_time), np.sum(all_chase_time), np.sum(all_end_chase_time),
                               np.sum(all_after_chase_time), np.sum(all_before_contact_time), np.sum(all_after_contact_time)]

        time_context_values.append(len(all_pre_chase_time) * 3*60*60 - np.sum(time_context_values))
        time_context_values /= np.sum(time_context_values)

        # fig, ax = plt.subplots(figsize=(12/2.54,12/2.54))
        size = 0.3
        outer_colors = ['tab:red', 'tab:orange', 'yellow', 'tab:green', 'k','tab:brown', 'tab:grey']
        ax_pie.pie(event_context_values, radius=1, colors=outer_colors,
               wedgeprops=dict(width=size, edgecolor='w'), startangle=90, center=(0, 1))
        ax_pie.pie(time_context_values, radius=1-size, colors=outer_colors,
               wedgeprops=dict(width=size, edgecolor='w', alpha=.6), startangle=90, center=(0, 1))

        ax_pie.set_title(r'event context')
        legend_elements = [Patch(facecolor='tab:red', edgecolor='w', label='%.1f' % (event_context_values[0] * 100) + '%'),
                           Patch(facecolor='tab:orange', edgecolor='w', label='%.1f' % (event_context_values[1] * 100) + '%'),
                           Patch(facecolor='yellow', edgecolor='w', label='%.1f' % (event_context_values[2] * 100) + '%'),
                           Patch(facecolor='tab:green', edgecolor='w', label='%.1f' % (event_context_values[3] * 100) + '%'),
                           Patch(facecolor='k', edgecolor='w', label='%.1f' % (event_context_values[4] * 100) + '%'),
                           Patch(facecolor='tab:brown', edgecolor='w', label='%.1f' % (event_context_values[5] * 100) + '%'),
                           Patch(facecolor='tab:red', alpha=0.6, edgecolor='w', label='%.1f' % (time_context_values[0] * 100) + '%'),
                           Patch(facecolor='tab:orange', alpha=0.6, edgecolor='w', label='%.1f' % (time_context_values[1] * 100) + '%'),
                           Patch(facecolor='yellow', alpha=0.6, edgecolor='w', label='%.1f' % (time_context_values[2] * 100) + '%'),
                           Patch(facecolor='tab:green', alpha=0.6, edgecolor='w', label='%.1f' % (time_context_values[3] * 100) + '%'),
                           Patch(facecolor='k', alpha=0.6, edgecolor='w', label='%.1f' % (time_context_values[4] * 100) + '%'),
                           Patch(facecolor='tab:brown', alpha=0.6, edgecolor='w', label='%.1f' % (time_context_values[5] * 100) + '%')]

        ax_pie.legend(handles=legend_elements, loc='lower right', ncol=2, bbox_to_anchor=(1.15, -0.25), frameon=False, fontsize=9)

        plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'event_time_corr', f'{event_name}_categories.png'), dpi=300)
        plt.close()
        # plt.show()



if __name__ == '__main__':
    main(sys.argv[1])
