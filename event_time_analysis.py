import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from IPython import embed
from event_time_correlations import load_and_converete_boris_events, kde, gauss


def main(base_path):
    trial_summary = pd.read_csv('trial_summary.csv', index_col=0)
    female_color, male_color = '#e74c3c', '#3498db'

    all_rise_times_lose = []
    all_rise_times_win = []
    all_chirp_times_lose = []
    all_chirp_times_win = []
    win_sex = []
    lose_sex = []

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


    embed()
    quit()
    ici_lose = []
    ici_win = []
    for i in range(len(all_chirp_times_lose)):
        ici_lose.append(np.diff(all_chirp_times_lose[i]))
        ici_win.append(np.diff(all_chirp_times_win[i]))

    fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
    ax = fig.add_subplot(gs[0, 0])

    for i in range(len(ici_lose)):
        if win_sex[i] == 'm':
            if lose_sex[i] == 'm':
                color, linestyle = male_color, '-'
            else:
                color, linestyle = male_color, '--'
        else:
            if lose_sex[i] == 'm':
                color, linestyle = female_color, '--'
            else:
                color, linestyle = female_color, '-'


        conv_y_chirp_lose = np.arange(0, 30, .5)
        kde_array = kde(ici_lose[i], conv_y_chirp_lose, kernal_w=1, kernal_h=1)

        # kde_array /= np.sum(kde_array)
        ax.plot(conv_y_chirp_lose, kde_array, zorder=2, color=color, linestyle=linestyle, lw=2)

    plt.show()

    pass


if __name__ == '__main__':
    main(sys.argv[1])
