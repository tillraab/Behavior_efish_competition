import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from IPython import embed
from event_time_correlations import load_and_converete_boris_events, kde, gauss

female_color, male_color = '#e74c3c', '#3498db'

def iei_analysis(all_chirp_times_lose, all_chirp_times_win, all_rise_times_lose, all_rise_times_win, win_sex, lose_sex):
    ici_lose = []
    ici_win = []

    iri_lose = []
    iri_win = []

    for i in range(len(all_chirp_times_lose)):
        ici_lose.append(np.diff(all_chirp_times_lose[i]))
        ici_win.append(np.diff(all_chirp_times_win[i]))

        iri_lose.append(np.diff(all_rise_times_lose[i]))
        iri_win.append(np.diff(all_rise_times_win[i]))

    for iei, kernal_w in zip([ici_lose, ici_win, iri_lose, iri_win],
                             [1, 1, 5, 50]):

        fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
        gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.95, top=0.95)
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


            conv_y_chirp_lose = np.arange(0, np.percentile(np.hstack(iei), 90), .5)
            kde_array = kde(iei[i], conv_y_chirp_lose, kernal_w=kernal_w, kernal_h=1)

            # kde_array /= np.sum(kde_array)
            ax[sp].plot(conv_y_chirp_lose, kde_array, zorder=2, color=color, linestyle=linestyle, lw=2)

        plt.setp(ax[1].get_yticklabels(), visible=False)
        plt.setp(ax[3].get_yticklabels(), visible=False)


        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_xticklabels(), visible=False)
        plt.show()


def relative_rate_progression(all_event_t, title=''):
    stop_t = 3*60*60
    snippet_len = 15*60

    snippet_starts = np.arange(0, stop_t, snippet_len)
    all_snippet_ratio = []
    for event_t in all_event_t:
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

    plt.show()


def main(base_path):
    trial_summary = pd.read_csv('trial_summary.csv', index_col=0)

    all_rise_times_lose = []
    all_rise_times_win = []
    all_chirp_times_lose = []
    all_chirp_times_win = []
    win_sex = []
    lose_sex = []

    all_contact_t = []
    all_ag_on_t = []

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

        ### communication
        if not os.path.exists(os.path.join(trial_path, 'chirp_times_cnn.npy')):
            continue
        chirp_t = np.load(os.path.join(trial_path, 'chirp_times_cnn.npy'))
        chirp_ids = np.load(os.path.join(trial_path, 'chirp_ids_cnn.npy'))
        chirp_times = [chirp_t[chirp_ids == trial['win_ID']], chirp_t[chirp_ids == trial['lose_ID']]]


        rise_idx = np.load(os.path.join(trial_path, 'analysis', 'rise_idx.npy'))[::sorter]
        rise_idx_int = [np.array(rise_idx[i][~np.isnan(rise_idx[i])], dtype=int) for i in range(len(rise_idx))]
        rise_times = [times[rise_idx_int[0]], times[rise_idx_int[1]]]

        # iri = np.diff(rise_times[1])
        all_rise_times_lose.append(rise_times[1])
        all_rise_times_win.append(rise_times[0])

        all_chirp_times_lose.append(chirp_times[1])
        all_chirp_times_win.append(chirp_times[0])

        win_sex.append(trial['sex_win'])
        lose_sex.append(trial['sex_lose'])

    iei_analysis(all_chirp_times_lose, all_chirp_times_win, all_rise_times_lose, all_rise_times_win, win_sex, lose_sex)

    relative_rate_progression(all_chirp_times_lose, title=r'chirp$_{lose}$')
    relative_rate_progression(all_chirp_times_win, title=r'chirp$_{win}$')
    relative_rate_progression(all_rise_times_lose, title=r'rise$_{lose}$')
    relative_rate_progression(all_rise_times_win, title=r'rise$_{win}$')

    relative_rate_progression(all_contact_t, title=r'contact')
    relative_rate_progression(all_ag_on_t, title=r'chasing')

    embed()
    quit()
    pass


if __name__ == '__main__':
    main(sys.argv[1])
