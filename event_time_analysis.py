import os
import sys
import argparse
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from IPython import embed


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

def kde(event_dt, max_dt = 60):
    kernal_w = 1
    kernal_h = 0.2

    conv_t = np.arange(-max_dt, max_dt, 1)
    conv_array = np.zeros(len(conv_t))

    for e in event_dt:
        conv_array += gauss(conv_t, e, kernal_w, kernal_h, norm=True)

    plt.plot(conv_t, conv_array)


def permulation_kde(event_dt, repetitions = 100, max_dt = 60):
    embed()
    quit()

    kernal_w = 1
    kernal_h = 0.2

    conv_t = cp.arange(-max_dt, max_dt, 1)
    conv_tt = cp.reshape(conv_t, (len(conv_t), 1, 1))

    # array.shape = (120, 100, 15486) = (len(conv_t), repetitions, len(event_dt))
    event_dt_perm = cp.tile(event_dt, (len(conv_t), repetitions, 1))
    # conv_t_perm = cp.tile(conv_tt, (1, repetitions, len(event_dt)))

    gauss_3d = cp.exp(-((conv_tt - event_dt_perm) / kernal_w) ** 2 / 2) * kernal_h
    kde_3d = cp.sum(gauss_3d, axis = 2).transpose()
    kde_3d_numpy = cp.asnumpy(kde_3d)


def main(base_path):
    trial_summary = pd.read_csv('trial_summary.csv', index_col=0)

    lose_chrips_centered_on_ag_off_t = []
    for index, trial in trial_summary.iterrows():
        trial_path = os.path.join(base_path, trial['recording'])

        if trial['group'] < 5:
            continue
        if not os.path.exists(os.path.join(trial_path, 'led_idxs.csv')):
            continue
        if not os.path.exists(os.path.join(trial_path, 'LED_frames.npy')):
            continue

        ids = np.load(os.path.join(trial_path, 'analysis', 'ids.npy'))
        times = np.load(os.path.join(trial_path, 'times.npy'))
        sorter = -1 if trial['win_ID'] != ids[0] else 1

        ### event times --> BORIS behavior
        contact_t_GRID, ag_on_off_t_GRID, led_idx, led_frames = \
            load_and_converete_boris_events(trial_path, trial['recording'], sr=20_000)

        ### communication
        got_chirps = False
        if os.path.exists(os.path.join(trial_path, 'chirp_times_cnn.npy')):
            chirp_t = np.load(os.path.join(trial_path, 'chirp_times_cnn.npy'))
            chirp_ids = np.load(os.path.join(trial_path, 'chirp_ids_cnn.npy'))
            chirp_times = [chirp_t[chirp_ids == trial['win_ID']], chirp_t[chirp_ids == trial['lose_ID']]]
            got_chirps = True

        rise_idx = np.load(os.path.join(trial_path, 'analysis', 'rise_idx.npy'))[::sorter]
        rise_idx_int = [np.array(rise_idx[i][~np.isnan(rise_idx[i])], dtype=int) for i in range(len(rise_idx))]
        rise_times = [times[rise_idx_int[0]], times[rise_idx_int[1]]]

        lose_chrips_centered_on_ag_off_t.append(event_centered_times(ag_on_off_t_GRID[:, 1], chirp_times[1]))

    kde(np.hstack(lose_chrips_centered_on_ag_off_t))

    permulation_kde(np.hstack(lose_chrips_centered_on_ag_off_t))


    embed()
    quit()
    pass

if __name__ == '__main__':
    main(sys.argv[1])