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

def get_baseline_freq(fund_v, idx_v, times, ident_v, idents = None, binwidth = 300):
    if not hasattr(idents, '__len__'):
        idents = np.unique(ident_v[~np.isnan(ident_v)])

    base_freqs = []
    for id in idents:
        f = fund_v[ident_v == id]
        t = times[idx_v[ident_v == id]]
        bins = np.arange(-binwidth/2, times[-1] + binwidth/2, binwidth)
        base_f = np.full(len(bins)-1, np.nan)
        for i in range(len(bins)-1):
            Cf = f[(t > bins[i]) & (t <= bins[i+1])]
            if len(Cf) == 0:
                continue
            else:
                base_f[i] = np.percentile(Cf, 5)
        base_freqs.append(base_f)

    return np.array(base_freqs), np.array(bins[:-1] + (bins[1] - bins[0])/2)

def frequency_q10_compensation(baseline_freq, temp, temp_t, light_start_sec):
    q10_comp_freq = []
    q10_vals = []
    for bf in baseline_freq:
        Cbf = np.copy(bf)
        Ctemp = np.copy(temp)
        if len(Cbf) > len(Ctemp):
            Cbf = Cbf[:len(Ctemp)]
        elif len(Ctemp) > len(Cbf):
            Ctemp = Ctemp[:len(Cbf)]
        else:
            pass
        q10s = []
        for i in range(len(Cbf) - 1):
            for j in np.arange(len(Cbf) - 1) + 1:
                if Cbf[i] == Cbf[j] or Ctemp[i] == Ctemp[j]:
                    # q10 with same values is useless
                    continue
                if temp_t[i] < light_start_sec or temp_t[j] < light_start_sec:
                    # to much frequency changes due to rises in first part of rec !!!
                    continue

                Cq10 = q10(Cbf[i], Cbf[j], Ctemp[i], Ctemp[j])
                q10s.append(Cq10)

        q10_comp_freq.append(Cbf * np.median(q10s) ** ((25 - Ctemp) / 10))
        q10_vals.append(np.median(q10s))

    return q10_comp_freq, q10_vals

def get_temperature(folder_path):
    temp_file = pd.read_csv(os.path.join(folder_path, 'temperatures.csv'), sep=';')
    temp_t = temp_file[temp_file.keys()[0]]
    temp = temp_file[temp_file.keys()[1]]

    temp_t = np.array(temp_t)
    temp = np.array(temp)


    if type(temp[-1]).__name__== 'str':
        temp = np.array(temp[:-1], dtype=float)
        temp_t = np.array(temp_t[:-1], dtype=int)

    return np.array(temp_t), np.array(temp)

def main(data_folder=None):
    colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']
    female_color, male_color = '#e74c3c', '#3498db'
    Wc, Lc = 'darkgreen', '#3673A4'

    trials_meta = pd.read_csv('order_meta.csv')
    fish_meta = pd.read_csv('id_meta.csv')
    fish_meta['mean_w'] = np.nanmean(fish_meta.loc[:, ['w1', 'w2', 'w3']], axis=1)
    fish_meta['mean_l'] = np.nanmean(fish_meta.loc[:, ['l1', 'l2', 'l3']], axis=1)

    video_stated_FPS = 25  # cap.get(cv2.CAP_PROP_FPS)
    sr = 20_000

    trial_summary = pd.DataFrame(columns=['sex_win', 'sex_lose', 'size_win', 'size_lose', 'EODf_win', 'EODf_lose',
                                          'exp_win', 'exp_lose', 'chirps_win', 'chirps_lose', 'rises_win', 'rise_lose'])
    trial_summary_row = {f'{s}':None for s in trial_summary.keys()}

    for trial_idx in range(len(trials_meta)):
        group = trials_meta['group'][trial_idx]
        recording = trials_meta['recording'][trial_idx][1:-1]
        rec_id1 = trials_meta['rec_id1'][trial_idx]
        rec_id2 = trials_meta['rec_id2'][trial_idx]

        if group < 3:
            continue

        trial_path = os.path.join(data_folder, recording)
        if not os.path.exists(trial_path):
            continue

        if not os.path.exists(os.path.join(trial_path, 'led_idxs.csv')):
            continue

        if not os.path.exists(os.path.join(trial_path, 'LED_frames.npy')):
            continue

        #############################################################################################################
        ### meta collect
        win_id = rec_id1 if trials_meta['fish1'][trial_idx] == trials_meta['winner'][trial_idx] else rec_id2
        lose_id = rec_id2 if trials_meta['fish1'][trial_idx] == trials_meta['winner'][trial_idx] else rec_id1

        f1_length = float(fish_meta['mean_l'][(fish_meta['group'] == trials_meta['group'][trial_idx]) &
                                              (fish_meta['fish'] == trials_meta['fish1'][trial_idx])])
        f2_length = float(fish_meta['mean_l'][(fish_meta['group'] == trials_meta['group'][trial_idx]) &
                                              (fish_meta['fish'] == trials_meta['fish2'][trial_idx])])

        win_l = f1_length if trials_meta['fish1'][trial_idx] == trials_meta['winner'][trial_idx] else f2_length
        lose_l = f2_length if trials_meta['fish1'][trial_idx] == trials_meta['winner'][trial_idx] else f1_length

        win_exp = trials_meta['exp1'][trial_idx] if trials_meta['winner'][trial_idx] == trials_meta['fish1'][trial_idx] else trials_meta['exp2'][trial_idx]
        lose_exp = trials_meta['exp2'][trial_idx] if trials_meta['winner'][trial_idx] == trials_meta['fish1'][trial_idx] else trials_meta['exp1'][trial_idx]
        #############################################################################################################

        fund_v = np.load(os.path.join(trial_path, 'fund_v.npy'))
        ident_v = np.load(os.path.join(trial_path, 'ident_v.npy'))
        idx_v = np.load(os.path.join(trial_path, 'idx_v.npy'))
        times = np.load(os.path.join(trial_path, 'times.npy'))

        print('')
        if len(uid:=np.unique(ident_v[~np.isnan(ident_v)])) >2:
            print(f'to many ids: {len(uid)}')
        print(f'ids in recording: {uid[0]:.0f} {uid[1]:.0f}')
        print(f'ids in meta: {rec_id1:.0f} {rec_id2:.0f}')

        meta_id_in_uid = list(map(lambda x: x in uid, [rec_id1, rec_id2]))
        if ~np.all(meta_id_in_uid):
            continue

        temp_t, temp = get_temperature(trial_path)

        #############################################################################################################
        ### communication
        got_chirps = False
        if os.path.exists(os.path.join(trial_path, 'chirp_times_cnn.npy')):
            chirp_t = np.load(os.path.join(trial_path, 'chirp_times_cnn.npy'))
            chirp_ids = np.load(os.path.join(trial_path, 'chirp_ids_cnn.npy'))
            got_chirps = True

        chirp_times = [chirp_t[chirp_ids == win_id], chirp_t[chirp_ids == lose_id]]
        rise_idx = np.load(os.path.join(trial_path, 'analysis', 'rise_idx.npy'))
        rise_idx_int = [np.array(rise_idx[i][~np.isnan(rise_idx[i])], dtype=int) for i in range(len(rise_idx))]

        #############################################################################################################
        ### physical behavior
        contact_t_GRID, ag_on_off_t_GRID, led_idx, led_frames = \
            load_and_converete_boris_events(trial_path, recording, sr, video_stated_FPS=video_stated_FPS)

        trial_summary.loc[len(trial_summary)] = trial_summary_row
        trial_summary.iloc[-1] = {'sex_win': 'n',
                                  'sex_lose': 'n',
                                  'size_win': win_l,
                                  'size_lose': lose_l,
                                  'EODf_win': -1,
                                  'EODf_lose': -1,
                                  'exp_win': win_exp,
                                  'exp_lose': lose_exp,
                                  'chirps_win': len(chirp_times[0]),
                                  'chirps_lose': len(chirp_times[1]),
                                  'rises_win': len(rise_idx_int[0]),
                                  'rise_lose': len(rise_idx_int[1])
                                  }
        # embed()

        ###############################################################################
        fig = plt.figure(figsize=(30/2.54, 18/2.54))
        gs = gridspec.GridSpec(2, 1, left = 0.1, bottom = 0.1, right=0.95, top=0.95, height_ratios=[1, 3], hspace=0)
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0]))

        ax[1].plot(times[idx_v[ident_v == win_id]] / 3600, fund_v[ident_v == win_id], color=Wc)
        ax[1].plot(times[idx_v[ident_v == lose_id]] / 3600, fund_v[ident_v == lose_id], color=Lc)

        ax[0].plot(contact_t_GRID / 3600, np.ones_like(contact_t_GRID) , '|', markersize=10, color='k')
        ax[0].plot(ag_on_off_t_GRID[:, 0] / 3600, np.ones_like(ag_on_off_t_GRID[:, 0]) * 2, '|', markersize=10, color='firebrick')

        ax[0].plot(times[rise_idx_int[0]] / 3600, np.ones_like(rise_idx_int[0]) * 4, '|', markersize=10, color=Wc)
        ax[0].plot(times[rise_idx_int[1]] / 3600, np.ones_like(rise_idx_int[1]) * 5, '|', markersize=10, color=Lc)

        if got_chirps:
            ax[0].plot(chirp_times[0] / 3600, np.ones_like(chirp_times[0]) * 7, '|', markersize=10, color=Wc)
            ax[0].plot(chirp_times[1] / 3600, np.ones_like(chirp_times[1]) * 8, '|', markersize=10, color=Lc)

        min_f, max_f = np.min(fund_v[~np.isnan(ident_v)]), np.nanmax(fund_v[~np.isnan(ident_v)])

        ax[0].set_ylim(0, 9)
        ax[0].set_yticks([1, 2, 4, 5, 7, 8])
        ax[0].set_yticklabels(['contact', 'chase', r'rise$_{win}$', r'rise$_{lose}$', r'chirp$_{win}$', r'chirp$_{lose}$'])
        ax[1].set_ylim(min_f-50, max_f+50)

        ax[1].set_xlim(times[0]/3600, times[-1]/3600)
        plt.setp(ax[0].get_xticklabels(), visible=False)

        fig.suptitle(f'{recording}')


        plt.show()


    fig = plt.figure(figsize=(20/2.54, 20/2.54))
    gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.95, top=0.95, height_ratios=[1, 3], width_ratios=[3, 1])
    ax = fig.add_subplot(gs[1, 0])

    ax.plot(trial_summary['rises_win'], trial_summary['chirps_win'], 'o', color=Wc, label='winner')
    ax.plot(trial_summary['rise_lose'], trial_summary['chirps_lose'], 'o', color=Lc, label='loster')
    ax.set_xlabel('rises [n]', fontsize=12)
    ax.set_ylabel('chirps [n]', fontsize=12)
    ax.tick_params(labelsize=10)

    ax_chirps = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_chirps.boxplot([trial_summary['chirps_win'], trial_summary['chirps_lose']], widths = .5, positions = [1, 2])
    ax_chirps.set_xticks([1, 2])
    ax_chirps.set_xticklabels(['Win', 'Lose'])
    plt.setp(ax_chirps.get_yticklabels(), visible=False)

    ax_rises = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_rises.boxplot([trial_summary['rises_win'], trial_summary['rise_lose']], widths = .5, positions = [1, 2], vert=False)
    ax_rises.set_yticks([1, 2])
    ax_rises.set_yticklabels(['Win', 'Lose'])
    plt.setp(ax_rises.get_xticklabels(), visible=False)

    plt.show()


    embed()
    quit()
    pass

if __name__ == '__main__':
    # main("/home/raab/data/mount_data/")
    main("/home/raab/data/2020_competition_mount")