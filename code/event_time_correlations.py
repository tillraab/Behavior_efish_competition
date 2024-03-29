import os
import sys
import argparse
import pathlib
import time
import itertools

import numpy as np
try:
    import cupy as cp
except ImportError:
    import numpy as cp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from IPython import embed
from tqdm import tqdm

female_color, male_color = '#e74c3c', '#3498db'


def load_and_converete_boris_events(trial_path, recording, sr):
    def converte_video_frames_to_grid_idx(event_frames, led_frames, led_idx):
        event_idx_grid = (event_frames - led_frames[0]) / (led_frames[-1] - led_frames[0]) * (led_idx[-1] - led_idx[0]) + led_idx[0]
        return event_idx_grid

    # idx in grid-recording
    led_idx = pd.read_csv(os.path.join(trial_path, 'led_idxs.csv'), header=None).iloc[:, 0].to_numpy()
    # frames where LED gets switched on
    led_frames = np.load(os.path.join(trial_path, 'LED_frames.npy'))

    times, behavior, t_ag_on_off, t_contact, video_FPS = load_boris(trial_path, recording)

    contact_frame = np.array(np.round(t_contact * video_FPS), dtype=int)
    ag_on_off_frame = np.array(np.round(t_ag_on_off * video_FPS), dtype=int)

    # led_t_GRID = led_idx / sr
    contact_t_GRID = converte_video_frames_to_grid_idx(contact_frame, led_frames, led_idx) / sr
    ag_on_off_t_GRID = converte_video_frames_to_grid_idx(ag_on_off_frame, led_frames, led_idx) / sr

    return contact_t_GRID, ag_on_off_t_GRID, led_idx, led_frames


def load_boris(trial_path, recording):
    boris_file = '-'.join(recording.split('-')[:3]) + '.csv'

    data = pd.read_csv(os.path.join(trial_path, boris_file))
    times = data['Start (s)']
    behavior = data['Behavior']

    t_ag_on = times[behavior == 0]
    t_ag_off = times[behavior == 1]

    t_ag_on_off = []
    for t in t_ag_on:
        t1 = np.array(t_ag_off)[t_ag_off > t]
        if len(t1) >= 1:
            t_ag_on_off.append(np.array([t, t1[0]]))

    t_contact = times[behavior == 2]

    return times, behavior, np.array(t_ag_on_off), t_contact.to_numpy(), data['FPS'][0]


def gauss(t, shift, sigma, size, norm = False):
    if not hasattr(shift, '__len__'):
        g = np.exp(-((t - shift) / sigma) ** 2 / 2) * size
        if norm:
            g /= np.sum(g)
        return g
    else:
        t = np.array([t, ] * len(shift))
        res = np.exp(-((t.transpose() - shift).transpose() / sigma) ** 2 / 2) * size
        return res


def event_centered_times(centered_event_times, surrounding_event_times, max_dt = np.inf):

    event_dt = []
    for Cevent_t in centered_event_times:
        Cdt = np.array(surrounding_event_times - Cevent_t)
        event_dt.extend(Cdt[np.abs(Cdt) <= max_dt])

    return np.array(event_dt)


def kde(event_dt, conv_t, kernal_w = 1, kernal_h = 0.2):
    conv_array = np.zeros(len(conv_t))
    for e in event_dt:
        conv_array += gauss(conv_t, e, kernal_w, kernal_h)
    return conv_array


def permutation_kde(event_dt, conv_t, repetitions = 2000, max_mem_use_GB = 2, kernal_w = 1, kernal_h = 0.2):
    def chunk_permutation(select_event_dt, conv_tt, n_chuck, max_jitter, kernal_w, kernal_h):
        # array.shape = (120, 100, 15486) = (len(conv_t), repetitions, len(event_dt))
        # event_dt_perm = cp.tile(event_dt, (len(conv_t), repetitions, 1))
        event_dt_perm = cp.tile(select_event_dt, (len(conv_tt), n_chuck, 1))
        jitter = cp.random.uniform(-max_jitter, max_jitter, size=(event_dt_perm.shape[1], event_dt_perm.shape[2]))
        jitter = cp.expand_dims(jitter, axis=0)

        event_dt_perm += jitter
        # conv_t_perm = cp.tile(conv_tt, (1, repetitions, len(event_dt)))

        gauss_3d = cp.exp(-((conv_tt - event_dt_perm) / kernal_w) ** 2 / 2) * kernal_h

        kde_3d = cp.sum(gauss_3d, axis = 2).transpose()

        try:
            kde_3d_numpy = cp.asnumpy(kde_3d)
            del event_dt_perm, gauss_3d, kde_3d
            return kde_3d_numpy

        except AttributeError:
            del event_dt_perm, gauss_3d
            return kde_3d

    t0 = time.time()

    max_jitter = float(2*cp.max(conv_t))
    select_event_dt = event_dt[np.abs(event_dt) <= float(cp.max(conv_t)) + max_jitter*2]

    # conv_t = cp.arange(-max_dt, max_dt, 1)
    conv_tt = cp.reshape(conv_t, (len(conv_t), 1, 1))

    chunk_size = int(np.floor(max_mem_use_GB / (select_event_dt.nbytes * conv_t.size / 1e9)))
    chunk_collector =[]

    for _ in range(repetitions // chunk_size):
        chunk_boot_KDE = chunk_permutation(select_event_dt, conv_tt, chunk_size, max_jitter, kernal_w, kernal_h)
        chunk_collector.extend(chunk_boot_KDE)
        # # array.shape = (120, 100, 15486) = (len(conv_t), repetitions, len(event_dt))
        # # event_dt_perm = cp.tile(event_dt, (len(conv_t), repetitions, 1))
        # event_dt_perm = cp.tile(event_dt, (len(conv_t), chunk_size, 1))
        # jitter = np.random.uniform(-max_jitter, max_jitter, size=(event_dt_perm.shape[1], event_dt_perm.shape[2]))
        # jitter = np.expand_dims(jitter, axis=0)
        #
        # event_dt_perm += jitter
        # # conv_t_perm = cp.tile(conv_tt, (1, repetitions, len(event_dt)))
        #
        # gauss_3d = cp.exp(-((conv_tt - event_dt_perm) / kernal_w) ** 2 / 2) * kernal_h
        # kde_3d = cp.sum(gauss_3d, axis = 2).transpose()
        # try:
        #     kde_3d_numpy = cp.asnumpy(kde_3d)
        #     chunk_collector.extend(kde_3d_numpy)
        # except AttributeError:
        #     chunk_collector.extend(kde_3d)
        # del event_dt_perm, gauss_3d, kde_3d
    chunk_boot_KDE = chunk_permutation(select_event_dt, conv_tt, repetitions % chunk_size, max_jitter, kernal_w, kernal_h)
    chunk_collector.extend(chunk_boot_KDE)
    chunk_collector = np.array(chunk_collector)

    # ToDo: this works but is incorrect i think
    # chunk_collector /= np.sum(chunk_collector, axis=1).reshape(chunk_collector.shape[0], 1)
    print(f'bootstrap with {repetitions:.0f} repetitions took {time.time() - t0:.2f}s.')

    # fig, ax = plt.subplots()
    # for i in range(len(chunk_collector)):
    #     ax.plot(cp.asnumpy(conv_t), chunk_collector[i])

    return chunk_collector


def jackknife_kde(event_dt, conv_t, repetitions = 2000, max_mem_use_GB = 2, jack_pct = 0.9, kernal_w = 1, kernal_h = 0.2):
    def chunk_jackknife(select_event_dt, conv_tt, n_chuck, jack_pct, kernal_w, kernal_h):
        event_dt_rep = cp.tile(select_event_dt, (n_chuck, 1))
        idx = cp.random.rand(*event_dt_rep.shape).argsort(1)[:, :int(event_dt_rep.shape[-1]*jack_pct)]
        event_dt_jk = event_dt_rep[cp.arange(event_dt_rep.shape[0])[:, None], idx]
        event_dt_jk_full = cp.tile(event_dt_jk, (len(conv_tt), 1, 1))

        gauss_3d = cp.exp(-((conv_tt - event_dt_jk_full) / kernal_w) ** 2 / 2) * kernal_h

        kde_3d = cp.sum(gauss_3d, axis = 2).transpose()

        try:
            kde_3d_numpy = cp.asnumpy(kde_3d)
            del event_dt_rep, idx, event_dt_jk, event_dt_jk_full, gauss_3d, kde_3d
            return kde_3d_numpy

        except AttributeError:
            del event_dt_rep, idx, event_dt_jk, event_dt_jk_full, gauss_3d
            return kde_3d

    t0 = time.time()

    # max_jitter = 2*max_dt
    select_event_dt = event_dt[np.abs(event_dt) <= float(cp.max(conv_t)) * 2]

    if len(select_event_dt) == 0:
        return np.zeros((repetitions, len(conv_t)))

    # conv_t = cp.arange(-max_dt, max_dt, 1)
    conv_tt = cp.reshape(conv_t, (len(conv_t), 1, 1))
    chunk_size = int(np.floor(max_mem_use_GB / (select_event_dt.nbytes * jack_pct * conv_t.size / 1e9)))

    chunk_collector =[]

    for _ in range(repetitions // chunk_size):
        chunk_jackknife_KDE = chunk_jackknife(select_event_dt, conv_tt, chunk_size, jack_pct, kernal_w, kernal_h)
        chunk_collector.extend(chunk_jackknife_KDE)
        del chunk_jackknife_KDE
        # # array.shape = (120, 100, 15486) = (len(conv_t), repetitions, len(event_dt))
        # # event_dt_perm = cp.tile(event_dt, (len(conv_t), repetitions, 1))
        # event_dt_perm = cp.tile(event_dt, (len(conv_t), chunk_size, 1))
        # jitter = np.random.uniform(-max_jitter, max_jitter, size=(event_dt_perm.shape[1], event_dt_perm.shape[2]))
        # jitter = np.expand_dims(jitter, axis=0)
        #
        # event_dt_perm += jitter
        # # conv_t_perm = cp.tile(conv_tt, (1, repetitions, len(event_dt)))
        #
        # gauss_3d = cp.exp(-((conv_tt - event_dt_perm) / kernal_w) ** 2 / 2) * kernal_h
        # kde_3d = cp.sum(gauss_3d, axis = 2).transpose()
        # try:
        #     kde_3d_numpy = cp.asnumpy(kde_3d)
        #     chunk_collector.extend(kde_3d_numpy)
        # except AttributeError:
        #     chunk_collector.extend(kde_3d)
        # del event_dt_perm, gauss_3d, kde_3d
    chunk_jackknife_KDE = chunk_jackknife(select_event_dt, conv_tt, repetitions % chunk_size, jack_pct, kernal_w, kernal_h)
    chunk_collector.extend(chunk_jackknife_KDE)
    del chunk_jackknife_KDE
    chunk_collector = np.array(chunk_collector)

    print(f'jackknife with {repetitions:.0f} repetitions took {time.time() - t0:.2f}s.')

    return chunk_collector


def single_kde(event_dt, conv_t, kernal_w = 1, kernal_h = 0.2):

    single_kdes = cp.zeros((len(event_dt), len(conv_t)))
    for enu, e_dt in enumerate(event_dt):
        Ce_dt = e_dt[np.abs(e_dt) <= float(cp.max(conv_t)) * 2]
        conv_tt = cp.reshape(conv_t, (len(conv_t), 1))
        Ce_dt_tile = cp.tile(Ce_dt, (len(conv_tt), 1))

        gauss_3d = cp.exp(-((conv_tt - Ce_dt_tile) / kernal_w) ** 2 / 2) * kernal_h
        single_kdes[enu] = cp.sum(gauss_3d, axis=1)


    return cp.asnumpy(single_kdes)


def main(base_path):
    if not os.path.exists(os.path.join(os.path.split(__file__)[0], 'figures', 'event_time_corr')):
        os.makedirs(os.path.join(os.path.split(__file__)[0], 'figures', 'event_time_corr'))

    trial_summary = pd.read_csv(os.path.join(base_path, 'trial_summary.csv'), index_col=0)
    chirp_notes = pd.read_csv(os.path.join(base_path, 'chirp_notes.csv'), index_col=0)
    # trial_summary = trial_summary[chirp_notes['good'] == 1]
    trial_mask = chirp_notes['good'] == 1

    # ToDo: do chirp on chirp and rise on rise
    lose_chrips_centered_on_ag_off_t = []
    lose_chrips_centered_on_ag_on_t = []
    lose_chrips_centered_on_contact_t = []
    lose_chrips_centered_on_win_rises = []
    lose_chrips_centered_on_win_chirp = []
    lose_chirps_centered_on_lose_rises = []

    win_chrips_centered_on_ag_off_t = []
    win_chrips_centered_on_ag_on_t = []
    win_chrips_centered_on_contact_t = []
    win_chrips_centered_on_lose_rises = []
    win_chrips_centered_on_lose_chirp = []
    win_chirps_centered_on_win_rises = []

    lose_rises_centered_on_ag_off_t = []
    lose_rises_centered_on_ag_on_t = []
    lose_rises_centered_on_contact_t = []
    lose_rises_centered_on_win_chirps = []

    win_rises_centered_on_ag_off_t = []
    win_rises_centered_on_ag_on_t = []
    win_rises_centered_on_contact_t = []
    win_rises_centered_on_lose_chirps = []

    ag_off_centered_on_ag_on = []

    lose_chirp_count = []
    win_chirp_count = []
    lose_rises_count = []
    win_rises_count = []
    chase_count = []
    contact_count = []

    sex_win = []
    sex_lose = []

    for index, trial in tqdm(trial_summary.iterrows()):
        trial_path = os.path.join(base_path, trial['recording'])

        if trial['group'] < 5:
            continue
        if not os.path.exists(os.path.join(trial_path, 'led_idxs.csv')):
            continue
        if not os.path.exists(os.path.join(trial_path, 'LED_frames.npy')):
            continue
        if trial['draw'] == 1:
            continue

        ids = np.load(os.path.join(trial_path, 'analysis', 'ids.npy'))
        times = np.load(os.path.join(trial_path, 'times.npy'))
        sorter = -1 if trial['win_ID'] != ids[0] else 1

        ### event times --> BORIS behavior
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

        ### collect for correlations ####
        # chirps
        if trial_mask[index]:
            lose_chrips_centered_on_ag_off_t.append(event_centered_times(ag_on_off_t_GRID[:, 1], chirp_times[1]))
            lose_chrips_centered_on_ag_on_t.append(event_centered_times(ag_on_off_t_GRID[:, 0], chirp_times[1]))
            lose_chrips_centered_on_contact_t.append(event_centered_times(contact_t_GRID, chirp_times[1]))
            lose_chrips_centered_on_win_rises.append(event_centered_times(rise_times[0], chirp_times[1]))
            lose_chrips_centered_on_win_chirp.append(event_centered_times(chirp_times[0], chirp_times[1]))
            lose_chirps_centered_on_lose_rises.append(event_centered_times(rise_times[1], chirp_times[1]))
            lose_chirp_count.append(len(chirp_times[1]))

            win_chrips_centered_on_ag_off_t.append(event_centered_times(ag_on_off_t_GRID[:, 1], chirp_times[0]))
            win_chrips_centered_on_ag_on_t.append(event_centered_times(ag_on_off_t_GRID[:, 0], chirp_times[0]))
            win_chrips_centered_on_contact_t.append(event_centered_times(contact_t_GRID, chirp_times[0]))
            win_chrips_centered_on_lose_rises.append(event_centered_times(rise_times[1], chirp_times[0]))
            win_chrips_centered_on_lose_chirp.append(event_centered_times(chirp_times[1], chirp_times[0]))
            win_chirps_centered_on_win_rises.append(event_centered_times(rise_times[0], chirp_times[0]))

            win_chirp_count.append(len(chirp_times[0]))
        else:
            lose_chrips_centered_on_ag_off_t.append(np.array([]))
            lose_chrips_centered_on_ag_on_t.append(np.array([]))
            lose_chrips_centered_on_contact_t.append(np.array([]))
            lose_chrips_centered_on_win_rises.append(np.array([]))
            lose_chrips_centered_on_win_chirp.append(np.array([]))
            lose_chirps_centered_on_lose_rises.append(np.array([]))
            lose_chirp_count.append(np.nan)

            win_chrips_centered_on_ag_off_t.append(np.array([]))
            win_chrips_centered_on_ag_on_t.append(np.array([]))
            win_chrips_centered_on_contact_t.append(np.array([]))
            win_chrips_centered_on_lose_rises.append(np.array([]))
            win_chrips_centered_on_lose_chirp.append(np.array([]))
            win_chirp_count.append(np.nan)

        # rises
        lose_rises_centered_on_ag_off_t.append(event_centered_times(ag_on_off_t_GRID[:, 1], rise_times[1]))
        lose_rises_centered_on_ag_on_t.append(event_centered_times(ag_on_off_t_GRID[:, 0], rise_times[1]))
        lose_rises_centered_on_contact_t.append(event_centered_times(contact_t_GRID, rise_times[1]))
        lose_rises_centered_on_win_chirps.append(event_centered_times(chirp_times[0], rise_times[1]))
        lose_rises_count.append(len(rise_times[1]))

        win_rises_centered_on_ag_off_t.append(event_centered_times(ag_on_off_t_GRID[:, 1], rise_times[0]))
        win_rises_centered_on_ag_on_t.append(event_centered_times(ag_on_off_t_GRID[:, 0], rise_times[0]))
        win_rises_centered_on_contact_t.append(event_centered_times(contact_t_GRID, rise_times[0]))
        win_rises_centered_on_lose_chirps.append(event_centered_times(chirp_times[1], rise_times[0]))
        win_rises_count.append(len(rise_times[0]))

        ag_off_centered_on_ag_on.append(event_centered_times(ag_on_off_t_GRID[:, 0], ag_on_off_t_GRID[:, 1]))
        chase_count.append(len(ag_on_off_t_GRID))
        contact_count.append(len(contact_t_GRID))

        sex_win.append(trial['sex_win'])
        sex_lose.append(trial['sex_lose'])

    sex_win = np.array(sex_win)
    sex_lose = np.array(sex_lose)
    # embed()
    # quit()
    max_dt = 30
    conv_t_dt = 0.5
    jack_pct = 0.9

    conv_t = cp.arange(-max_dt, max_dt+conv_t_dt, conv_t_dt)
    conv_t_numpy = cp.asnumpy(conv_t)

    for category_enu, (centered_times, event_counts, title) in enumerate(
            [[lose_chrips_centered_on_ag_off_t, chase_count, r'chirp$_{lose}$ on chase$_{off}$'],
             [lose_chrips_centered_on_ag_on_t, chase_count, r'chirp$_{lose}$ on chase$_{on}$'],
             [lose_chrips_centered_on_contact_t, contact_count, r'chirp$_{lose}$ on contact'],
             [lose_chrips_centered_on_win_rises, win_rises_count, r'chirp$_{lose}$ on rise$_{win}$'],
             [lose_chrips_centered_on_win_chirp, win_chirp_count, r'chirp$_{lose}$ on chirp$_{win}$'],
             [lose_chirps_centered_on_lose_rises, lose_rises_count, r'chirp$_{lose}$ on rises$_{lose}$'],

             [win_chrips_centered_on_ag_off_t, chase_count, r'chirp$_{win}$ on chase$_{off}$'],
             [win_chrips_centered_on_ag_on_t, chase_count, r'chirp$_{win}$ on chase$_{on}$'],
             [win_chrips_centered_on_contact_t, contact_count, r'chirp$_{win}$ on contact'],
             [win_chrips_centered_on_lose_rises, lose_rises_count, r'chirp$_{win}$ on rise$_{lose}$'],
             [win_chrips_centered_on_lose_chirp, lose_chirp_count, r'chirp$_{win}$ on chirp$_{lose}$'],
             [win_chirps_centered_on_win_rises, win_rises_count, r'chirp$_{win}$ on rises$_{win}$'],

             [lose_rises_centered_on_ag_off_t, chase_count, r'rise$_{lose}$ on chase$_{off}$'],
             [lose_rises_centered_on_ag_on_t, chase_count, r'rise$_{lose}$ on chase$_{on}$'],
             [lose_rises_centered_on_contact_t, contact_count, r'rise$_{lose}$ on contact'],
             [lose_rises_centered_on_win_chirps, win_chirp_count, r'rise$_{lose}$ on chirp$_{win}$'],

             [win_rises_centered_on_ag_off_t, chase_count, r'rise$_{win}$ on chase$_{off}$'],
             [win_rises_centered_on_ag_on_t, chase_count, r'rise$_{win}$ on chase$_{on}$'],
             [win_rises_centered_on_contact_t, contact_count, r'rise$_{win}$ on contact'],
             [win_rises_centered_on_lose_chirps, lose_chirp_count, r'rise$_{win}$ on chirp$_{lose}$'],

             [ag_off_centered_on_ag_on, chase_count, r'chase$_{off}$ on chase$_{on}$']]):

        save_str = title.replace('$', '').replace('{', '').replace('}', '').replace(' ', '_')

        ###########################################################################################################
        ### by pairing ###
        if True:
            centered_times_pairing = []
            for sex_w, sex_l in itertools.product(['m', 'f'], repeat=2):
                centered_times_pairing.append([])
                for i in range(len(centered_times)):
                    if sex_w == sex_win[i] and sex_l == sex_lose[i]:
                        centered_times_pairing[-1].append(centered_times[i])

            event_counts_pairings = [np.nansum(np.array(event_counts)[(sex_win == 'm') & (sex_lose == 'm')]),
                                     np.nansum(np.array(event_counts)[(sex_win == 'm') & (sex_lose == 'f')]),
                                     np.nansum(np.array(event_counts)[(sex_win == 'f') & (sex_lose == 'm')]),
                                     np.nansum(np.array(event_counts)[(sex_win == 'f') & (sex_lose == 'f')])]
            color = [male_color, female_color, male_color, female_color]
            linestyle = ['-', '--', '--', '-']

            perm_p_pairings = []
            jk_p_pairings = []
            fig = plt.figure(figsize=(20/2.54, 12/2.54))
            gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.95, top=0.9)
            ax = []
            ax.append(fig.add_subplot(gs[0, 0]))
            ax.append(fig.add_subplot(gs[0, 1], sharey=ax[0]))
            ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0]))
            ax.append(fig.add_subplot(gs[1, 1], sharey=ax[2], sharex=ax[1]))

            for enu, (centered_times_p, event_count_p) in enumerate(zip(centered_times_pairing, event_counts_pairings)):
                boot_kde = permutation_kde(np.hstack(centered_times_p), conv_t, kernal_w=1, kernal_h=1)
                jk_kde = jackknife_kde(np.hstack(centered_times_p), conv_t, jack_pct=jack_pct, kernal_w=1, kernal_h=1)

                perm_p1, perm_p50, perm_p99 = np.percentile(boot_kde, (1, 50, 99), axis=0)
                perm_p_pairings.append([perm_p1, perm_p50, perm_p99])

                jk_p1, jk_p50, jk_p99 = np.percentile(jk_kde, (1, 50, 99), axis=0)
                jk_p_pairings.append([jk_p1, jk_p50, jk_p99])

                ax[enu].fill_between(conv_t_numpy, perm_p1 / event_count_p, perm_p99 / event_count_p, color='cornflowerblue', alpha=.8)
                ax[enu].plot(conv_t_numpy, perm_p50 / event_count_p, color='dodgerblue', alpha=1, lw=3)

                ax[enu].fill_between(conv_t_numpy, jk_p1 / event_count_p / jack_pct, jk_p99 / event_count_p / jack_pct, color=color[enu], alpha=.8)
                ax[enu].plot(conv_t_numpy, jk_p50 / event_count_p / jack_pct, color=color[enu], alpha=1, lw=3, linestyle=linestyle[enu])

                ax_m = ax[enu].twinx()
                counter = 0
                for enu2, centered_events in enumerate(centered_times_p):
                    Cevents = centered_events[np.abs(centered_events) <= max_dt]
                    if len(Cevents) != 0:
                        ax_m.plot(Cevents, np.ones(len(Cevents)) * counter, '|', markersize=8, color='k', alpha=.1)
                        counter += 1

                ax_m.set_yticks([])
                ax[enu].set_xlim(-max_dt, max_dt)
                ax[enu].tick_params(labelsize=10)

            plt.setp(ax[1].get_yticklabels(), visible=False)
            plt.setp(ax[3].get_yticklabels(), visible=False)

            plt.setp(ax[0].get_xticklabels(), visible=False)
            plt.setp(ax[1].get_xticklabels(), visible=False)

            ax[2].set_xlabel('time [s]', fontsize=12)
            ax[3].set_xlabel('time [s]', fontsize=12)
            ax[0].set_ylabel('event rate [Hz]', fontsize=12)
            ax[2].set_ylabel('event rate [Hz]', fontsize=12)
            fig.suptitle(title)

            plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'event_time_corr', f'{save_str}_by_sexes.png'), dpi=300)
            plt.close()

        ###########################################################################################################
        ### all pairings ###
        boot_kde = permutation_kde(np.hstack(centered_times), conv_t, kernal_w=1, kernal_h=1)
        jk_kde = jackknife_kde(np.hstack(centered_times), conv_t, jack_pct=jack_pct, kernal_w=1, kernal_h=1)

        perm_p1, perm_p50, perm_p99 = np.percentile(boot_kde, (1, 50, 99), axis=0)
        jk_p1, jk_p50, jk_p99 = np.percentile(jk_kde, (1, 50, 99), axis=0)

        if category_enu == 0:
            # chirp on chase off
            chirp_on_chase_off = [perm_p1, perm_p50, perm_p99, jk_p1, jk_p50, jk_p99, event_counts]
        elif category_enu == 1:
            # chirp on chase on
            chirp_on_chase_on = [perm_p1, perm_p50, perm_p99, jk_p1, jk_p50, jk_p99, event_counts]
        elif category_enu == 20:
            # chase off on chase on
            chase_off_on_chase_on = [perm_p1, perm_p50, perm_p99, jk_p1, jk_p50, jk_p99, event_counts]

        elif category_enu == 12:
            rise_on_chase_off = [perm_p1, perm_p50, perm_p99, jk_p1, jk_p50, jk_p99, event_counts]
        elif category_enu == 13:
            rise_on_chase_on = [perm_p1, perm_p50, perm_p99, jk_p1, jk_p50, jk_p99, event_counts]

        elif category_enu == 2:
            chirp_on_contact = [perm_p1, perm_p50, perm_p99, jk_p1, jk_p50, jk_p99, event_counts]
        elif category_enu == 14:
            rise_on_contact = [perm_p1, perm_p50, perm_p99, jk_p1, jk_p50, jk_p99, event_counts]

        fig = plt.figure(figsize=(20/2.54, 12/2.54))
        gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
        ax = fig.add_subplot(gs[0, 0])

        ax.fill_between(conv_t_numpy, perm_p1/np.nansum(event_counts), perm_p99/np.nansum(event_counts), color='cornflowerblue', alpha=.8)
        ax.plot(conv_t_numpy, perm_p50/np.nansum(event_counts), color='dodgerblue', alpha=1, lw=3)

        ax.fill_between(conv_t_numpy, jk_p1/np.nansum(event_counts)/jack_pct, jk_p99/np.nansum(event_counts)/jack_pct, color='tab:red', alpha=.8)
        ax.plot(conv_t_numpy, jk_p50/np.nansum(event_counts)/jack_pct, color='firebrick', alpha=1, lw=3)

        ax_m = ax.twinx()
        counter = 0
        for enu, centered_events in enumerate(centered_times):
            Cevents = centered_events[np.abs(centered_events) <= max_dt]
            if len(Cevents) != 0:
                ax_m.plot(Cevents, np.ones(len(Cevents)) * counter, '|', markersize=8, color='k', alpha=.1)
                counter += 1
            # ax_m.plot(Cevents, np.ones(len(Cevents)) * enu, '|', markersize=8, color='k', alpha=.1)

        ax_m.set_yticks([])
        ax.set_xlabel('time [s]', fontsize=12)
        ax.set_ylabel('event rate [Hz]', fontsize=12)
        ax.set_title(title)
        ax.set_xlim(-max_dt, max_dt)
        ax.tick_params(labelsize=10)

        plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'event_time_corr', f'{save_str}.png'), dpi=300)
        plt.close()

    embed()
    quit()
    event_time_plot_with_agonistic_dur(conv_t_numpy, chirp_on_chase_off, chase_off_on_chase_on,
                                       lose_chrips_centered_on_ag_off_t, jack_pct, max_dt,
                                       title=r'chirp$_{lose}$ on chase$_{off}$', chase_on_centered=False)

    event_time_plot_with_agonistic_dur(conv_t_numpy, chirp_on_chase_on, chase_off_on_chase_on,
                                       lose_chrips_centered_on_ag_on_t, jack_pct, max_dt,
                                       title=r'chirp$_{lose}$ on chase$_{on}$', chase_on_centered=True)


    event_time_plot_with_agonistic_dur(conv_t_numpy, rise_on_chase_off, chase_off_on_chase_on,
                                       lose_rises_centered_on_ag_off_t, jack_pct, max_dt,
                                       title=r'rise$_{lose}$ on chase$_{off}$', chase_on_centered=False)

    event_time_plot_with_agonistic_dur(conv_t_numpy, rise_on_chase_on, chase_off_on_chase_on,
                                       lose_rises_centered_on_ag_on_t, jack_pct, max_dt,
                                       title=r'rise$_{lose}$ on chase$_{on}$', chase_on_centered=True)


    event_time_plot_with_agonistic_dur(conv_t_numpy, rise_on_contact, None,
                                       lose_rises_centered_on_contact_t, jack_pct, max_dt,
                                       title=r'rise$_{lose}$ on contact')

    event_time_plot_with_agonistic_dur(conv_t_numpy, chirp_on_contact, None,
                                       lose_chrips_centered_on_contact_t, jack_pct, max_dt,
                                       title=r'chirp$_{lose}$ on contact')





def event_time_plot_with_agonistic_dur(conv_t_numpy, centered_communication, chase_dur_dist,
                                       centered_raster_times, jack_pct, max_dt, title='', chase_on_centered=True):
    fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
    # fig = plt.figure(figsize=(14 / 2.54, 8 / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.125 , right=0.9, top=0.95)
    ax = fig.add_subplot(gs[0, 0])
    perm_p1, perm_p50, perm_p99, jk_p1, jk_p50, jk_p99, event_counts = centered_communication

    ax.fill_between(conv_t_numpy, perm_p1 / np.nansum(event_counts), perm_p99 / np.nansum(event_counts),
                    color='tab:gray', alpha=.8)
    ax.plot(conv_t_numpy, perm_p50 / np.nansum(event_counts), color='tab:gray', alpha=1, lw=3)

    ax.fill_between(conv_t_numpy, jk_p1 / np.nansum(event_counts) / jack_pct,
                    jk_p99 / np.nansum(event_counts) / jack_pct, color='tab:red', alpha=.8)
    ax.plot(conv_t_numpy, jk_p50 / np.nansum(event_counts) / jack_pct, color='firebrick', alpha=1, lw=3)

    ax.set_xlabel('time [s]', fontsize=14)
    ax.set_ylabel('event rate [Hz]', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-max_dt, max_dt)
    ax.tick_params(labelsize=12)

    ### chasing dist
    if hasattr(chase_dur_dist, '__len__'):
        ax_m = ax.twinx()
        perm_p1, perm_p50, perm_p99, jk_p1, jk_p50, jk_p99, event_counts = chase_dur_dist
        if chase_on_centered == False:
            y1label = r'p(chase$_{start}$)'
            jk_p1 = jk_p1[::-1]
            jk_p50 = jk_p50[::-1]
            jk_p99 = jk_p99[::-1]
            mask = conv_t_numpy <= 0
        else:
            y1label = r'p(chase$_{end}$)'
            mask = conv_t_numpy >= 0

        ax_m.fill_between(conv_t_numpy[mask], jk_p1[mask] / np.nansum(event_counts) / jack_pct,
                        jk_p99[mask] / np.nansum(event_counts) / jack_pct, color='tab:blue', alpha=.6)
        ax_m.plot(conv_t_numpy[mask], jk_p50[mask] / np.nansum(event_counts) / jack_pct, color='tab:blue', alpha=.75, lw=3)


        ax_m.set_ylabel(y1label, fontsize=14)
        ax_m.yaxis.label.set_color('tab:blue')
        ax_m.spines["right"].set_edgecolor('tab:blue')
        ax_m.tick_params(axis='y', colors='tab:blue', labelsize=12)

    counter = 0
    ax_m2 = ax.twinx()
    for enu, centered_events in enumerate(centered_raster_times):
        Cevents = centered_events[np.abs(centered_events) <= max_dt]
        if len(Cevents) != 0:
            ax_m2.plot(Cevents, np.ones(len(Cevents)) * counter, '|', markersize=8, color='k', alpha=.1)
            counter += 1

    ax_m2.plot([0, 0], [-1, counter], '--', color='k', lw=2)
    ax_m2.set_ylim(-1, counter)

    ax_m2.yaxis.set_visible(False)
    # embed()
    # quit()
    save_path = pathlib.Path(__file__).parent / 'figures' / 'event_time_corr_new'
    save_str = title.replace(' ', '_').replace('{', '').replace('}', '').replace('$', '')
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f'xx_{save_str}.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main(sys.argv[1])
