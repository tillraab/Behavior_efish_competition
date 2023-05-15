import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import sys
import glob
from IPython import embed


def load_frame_times(trial_path):
    t_filepath = glob.glob(os.path.join(trial_path, '*.dat'))
    if len(t_filepath) == 0:
        return np.array([])
    else:
        t_filepath = t_filepath[0]
    f = open(t_filepath, 'r')
    frame_t = []
    for line in f.readlines():
        t = sum(x * float(t) for x, t in zip([3600, 60, 1], line.replace('\n', '').split(":")))
        frame_t.append(t)
    return np.array(frame_t)


def load_and_converete_boris_events(trial_path, recording, sr, video_stated_FPS=25):
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


def main(data_folder=None):

    trials_meta = pd.read_csv('order_meta.csv')
    fish_meta = pd.read_csv('id_meta.csv')
    fish_meta['mean_w'] = np.nanmean(fish_meta.loc[:, ['w1', 'w2', 'w3']], axis=1)
    fish_meta['mean_l'] = np.nanmean(fish_meta.loc[:, ['l1', 'l2', 'l3']], axis=1)


    video_stated_FPS = 25  # cap.get(cv2.CAP_PROP_FPS)

    sr = 20_000

    for trial_idx in range(len(trials_meta)):
        print('')

        group = trials_meta['group'][trial_idx]
        recording = trials_meta['recording'][trial_idx][1:-1]
        rec_id1 = trials_meta['rec_id1'][trial_idx]
        rec_id2 = trials_meta['rec_id2'][trial_idx]

        f1_length = float(fish_meta['mean_l'][(fish_meta['group'] == trials_meta['group'][trial_idx]) &
                                              (fish_meta['fish'] == trials_meta['fish1'][trial_idx])])
        f2_length = float(fish_meta['mean_l'][(fish_meta['group'] == trials_meta['group'][trial_idx]) &
                                              (fish_meta['fish'] == trials_meta['fish2'][trial_idx])])

        if group < 3:
            continue

        trial_path = os.path.join(data_folder, recording)
        if not os.path.exists(trial_path):
            continue

        if not os.path.exists(os.path.join(trial_path, 'led_idxs.csv')):
            continue

        if not os.path.exists(os.path.join(trial_path, 'LED_frames.npy')):
            continue

        fund_v = np.load(os.path.join(trial_path, 'fund_v.npy'))
        ident_v = np.load(os.path.join(trial_path, 'ident_v.npy'))
        idx_v = np.load(os.path.join(trial_path, 'idx_v.npy'))
        times = np.load(os.path.join(trial_path, 'times.npy'))

        if len(uid:=np.unique(ident_v[~np.isnan(ident_v)])) >2:
            print(f'to many ids: {len(uid)}')
        print(f'ids in recording: {uid[0]:.0f} {uid[1]:.0f}')
        print(f'ids in meta: {rec_id1:.0f} {rec_id2:.0f}')

        meta_id_in_uid = list(map(lambda x: x in uid, [rec_id1, rec_id2]))
        if ~np.all(meta_id_in_uid):
            continue

        contact_t_GRID, ag_on_off_t_GRID, led_idx, led_frames = \
            load_and_converete_boris_events(trial_path, recording, sr, video_stated_FPS=video_stated_FPS)

        embed()
        quit()
        ###############################################################################
        fig = plt.figure(figsize=(30/2.54, 18/2.54))
        gs = gridspec.GridSpec(2, 1, left = 0.1, bottom = 0.1, right=0.95, top=0.95, height_ratios=[1, 3])
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0]))
        for id in uid:
            ax[1].plot(times[idx_v[ident_v == id]] / 3600, fund_v[ident_v == id], marker='.')

        ax[0].plot(contact_t_GRID / 3600, np.ones_like(contact_t_GRID) , '|', markersize=20, color='k')
        ax[0].plot(ag_on_off_t_GRID[:, 0] / 3600, np.ones_like(ag_on_off_t_GRID[:, 0]) * 2, '|', markersize=20, color='red')
        min_f, max_f = np.min(fund_v[~np.isnan(ident_v)]), np.nanmax(fund_v[~np.isnan(ident_v)])

        ax[0].set_ylim(0, 3)
        ax[0].set_yticks([1, 2])
        ax[0].set_yticklabels(['contact', 'chase'])
        ax[1].set_ylim(min_f-50, max_f+50)

        ax[1].set_xlim(times[0]/3600, times[-1]/3600)
        plt.setp(ax[0].get_xticklabels(), visible=False)

        fig.suptitle(f'{recording}')


        plt.show()



    embed()
    quit()
    pass

if __name__ == '__main__':
    # main("/home/raab/data/mount_data/")
    main("/home/raab/data/2020_competition_mount")