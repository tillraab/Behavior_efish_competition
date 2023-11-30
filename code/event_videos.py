import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from matplotlib.patches import patch
import os
import sys
import cv2
import glob
import argparse
from IPython import embed
from tqdm import tqdm
from thunderfish.powerspectrum import decibel
import pathlib
import pandas as pd

def main(folder, dt):
    video_path = glob.glob(os.path.join(folder, '2022*.mp4'))[0]
    create_video_path = os.path.join(folder, 'rise_video')
    if not os.path.exists(create_video_path):
        os.mkdir(create_video_path)


    video = cv2.VideoCapture(video_path) #  was 'cap'

    # fish_freqs = np.load(os.path.join(folder, 'analysis', 'fish_freq_interp.npy'))
    fish_freqs = np.load(os.path.join(folder, 'analysis', 'fish_freq.npy'))

    # rise_idx = np.load(os.path.join(folder, 'analysis', 'rise_idx.npy'))
    meta = pd.read_csv(pathlib.Path(folder).parent / 'meta.csv', sep=',', encoding='utf-7', index_col=0)
    filename = pathlib.Path(folder).name
    win_id = meta.loc[filename, 'Win_ID']
    lose_id = meta.loc[filename, 'Lose_ID']
    rise_bboxes = pd.read_csv(pathlib.Path(folder) / "risedetector_bboxes.csv", sep=',')
    chirp_bboxes = pd.read_csv(pathlib.Path(folder) / "chirpdetector_bboxes.csv", sep=',')
    rise_times = rise_bboxes['t0'][rise_bboxes['id'] == lose_id].to_numpy()
    chirp_times = chirp_bboxes['chirp_times'][(chirp_bboxes['assigned_track'] == lose_id)].to_numpy()

    # ToDo: rise and chipt times to times idxs!!!
    # embed()
    # quit()

    frame_times = np.load(os.path.join(folder, 'analysis', 'frame_times.npy'))

    times = np.load(os.path.join(folder, 'times.npy'))
    fill_freqs = np.load(os.path.join(folder, 'fill_freqs.npy'))
    fill_times = np.load(os.path.join(folder, 'fill_times.npy'))
    fill_spec_shape = np.load(os.path.join(folder, 'fill_spec_shape.npy'))
    fill_spec = np.memmap(os.path.join(folder, 'fill_spec.npy'), dtype='float', mode='r',
                               shape=(fill_spec_shape[0], fill_spec_shape[1]), order='F')

    for rise_time in rise_bboxes['t0'][rise_bboxes['id'] == lose_id].to_numpy():
        relevant_chirps = chirp_times[((chirp_times - rise_time) > 0 ) &
                                      ((chirp_times - rise_time) < dt * 3)]
        if len(relevant_chirps) == 0:
            continue

        rel_chirp_time = relevant_chirps - rise_time

        HH = int((rise_time / 3600) // 1)
        MM = int((rise_time - HH * 3600) // 60)
        SS = int(rise_time - HH * 3600 - MM * 60)

        frames_oi = np.arange(len(frame_times))[((frame_times - rise_time) >= -dt) & ((frame_times - rise_time) <= 3*dt)]
        idxs_oi = np.arange(len(times))[((times - rise_time) >= -dt) & ((times - rise_time) <= 3*dt)]

        fig = plt.figure(figsize=(16 * 2 / 2.54, 9 * 2 / 2.54))
        fig.patch.set_facecolor('black')
        gs = gridspec.GridSpec(6, 2, left=0.075, bottom=0.05, right=.99, top=0.95, width_ratios=(1.5, 3), hspace=.3,
                               wspace=0.05)
        ax = []
        ax.append(fig.add_subplot(gs[:, 1]))
        ax.append(fig.add_subplot(gs[1:3, 0]))
        ax.append(fig.add_subplot(gs[3:5, 0]))

        y00, y01 = np.nanmin(fish_freqs[0][idxs_oi]), np.nanmax(fish_freqs[0][idxs_oi])
        y10, y11 = np.nanmin(fish_freqs[1][idxs_oi]), np.nanmax(fish_freqs[1][idxs_oi])

        if y01 - y00 < 20:
            y01 = y00 + 20
        if y11 - y10 < 20:
            y11 = y10 + 20
        freq_span1 = (y01) - (y00)
        freq_span2 = (y11) - (y10)
        yspan = freq_span1 if freq_span1 > freq_span2 else freq_span2

        ax[1].plot(times[idxs_oi] - rise_time, fish_freqs[0][idxs_oi], marker='.', markersize=4, color='darkorange', lw=2,
                   alpha=0.4)
        ax[2].plot(times[idxs_oi] - rise_time, fish_freqs[1][idxs_oi], marker='.', markersize=4, color='forestgreen',
                   lw=2, alpha=0.4)
        ax[1].plot([0, 0], [y00 - yspan * 0.2, y00 + yspan * 1.3], '--', color='white')
        ax[2].plot([0, 0], [y10 - yspan * 0.2, y10 + yspan * 1.3], '--', color='white')

        for ct in rel_chirp_time:
            ax[2].plot([ct, ct], [y10 - yspan * 0.2, y10 + yspan * 1.3], '--', color='tab:orange')

        ax[1].set_xticks([-30, -15, 0, 15, 30])
        ax[2].set_xticks([-30, -15, 0, 15, 30])
        plt.setp(ax[1].get_xticklabels(), visible=False)

        # spectrograms
        f_mask1 = np.arange(len(fill_freqs))[(fill_freqs >= y00 - yspan * 0.2) & (fill_freqs <= y00 + yspan * 1.3)]
        f_mask2 = np.arange(len(fill_freqs))[(fill_freqs >= y10 - yspan * 0.2) & (fill_freqs <= y10 + yspan * 1.3)]
        t_mask = np.arange(len(fill_times))[(fill_times >= rise_time - dt * 4) & (fill_times <= rise_time + dt * 4)]

        ax[1].imshow(decibel(fill_spec[f_mask1[0]:f_mask1[-1], t_mask[0]:t_mask[-1]][::-1]),
                     extent=[-dt * 4, dt * 4, y00 - yspan * 0.2, y00 + yspan * 1.3],
                     aspect='auto', vmin=-100, vmax=-50, cmap='afmhot', interpolation='gaussian')
        ax[2].imshow(decibel(fill_spec[f_mask2[0]:f_mask2[-1], t_mask[0]:t_mask[-1]][::-1]),
                     extent=[-dt * 4, dt * 4, y10 - yspan * 0.2, y10 + yspan * 1.3],
                     aspect='auto', vmin=-100, vmax=-50, cmap='afmhot', interpolation='gaussian')

        ax[1].set_ylim(y00 - yspan * 0.1, y00 + yspan * 1.2)
        # ax[1].set_xlim(-dt * 3, dt * 3)
        ax[1].set_xlim(-dt, dt * 3)
        ax[2].set_ylim(y10 - yspan * 0.1, y10 + yspan * 1.2)
        # ax[2].set_xlim(-dt * 3, dt * 3)
        ax[2].set_xlim(-dt, dt * 3)

        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].tick_params(labelsize=12, color='white', labelcolor='white')
        ax[2].tick_params(labelsize=12, color='white', labelcolor='white')

        # embed()
        # quit()
        for a in ax[1:]:
            a.spines['left'].set_edgecolor('white')
            a.spines['bottom'].set_edgecolor('white')
        # for spine in ax[1].spines.values():
        #     spine.set_edgecolor('white')
        # for spine in ax[2].spines.values():
        #     spine.set_edgecolor('white')

        ax[2].set_xlabel('time [s]', fontsize=14, color='white')
        fig.text(0.02, 0.5, 'frequency [Hz]', fontsize=14, va='center', rotation='vertical', color='white')
        # embed()
        # quit()
        rise_dot_counter = 0
        rise_passed = False
        chirp_dot_counter = np.zeros(len(relevant_chirps), dtype=int)
        chirp_passed = np.zeros(len(relevant_chirps), dtype=bool)
        chirp_dot_handls = []
        chirp_active = False

        # plt.show()

        # embed()
        # quit()
        for i in tqdm(np.arange(len(frames_oi))):
            # break
            video.set(cv2.CAP_PROP_POS_FRAMES, int(frames_oi[i]))
            ret, frame = video.read()

            if i == 0:
                img = ax[0].imshow(frame)
                line1, = ax[1].plot([frame_times[frames_oi[i]] - rise_time, frame_times[frames_oi[i]] - rise_time],
                                    [y00 - yspan * 0.15, y00 + yspan * 1.3],
                                    color='white', lw=1)
                line2, = ax[2].plot([frame_times[frames_oi[i]] - rise_time, frame_times[frames_oi[i]] - rise_time],
                                    [y10 - yspan * 0.15, y10 + yspan * 1.3],
                                    color='white', lw=1)
            else:
                img.set_data(frame)
                line1.set_data([frame_times[frames_oi[i]] - rise_time, frame_times[frames_oi[i]] - rise_time],
                               [y00 - yspan * 0.15, y00 + yspan * 1.3])
                line2.set_data([frame_times[frames_oi[i]] - rise_time, frame_times[frames_oi[i]] - rise_time],
                               [y10 - yspan * 0.15, y10 + yspan * 1.3])

            if frame_times[frames_oi[i]] - rise_time > 0:
                if rise_passed == False:
                    rise_dot, = ax[0].plot(0.05, 0.95, 'o', color='white', transform=ax[0].transAxes, markersize=20)
                    rise_passed = True
                    for pos in ['left', 'bottom', 'right', 'top']:
                        ax[0].spines[pos].set_edgecolor('white')
                        ax[0].spines[pos].set_edgecolor('white')

                if rise_passed == True:
                    if rise_dot_counter < 6:
                        rise_dot_counter += 1
                    elif rise_dot_counter == 6:
                        rise_dot.remove()
                        if not chirp_active:
                            for pos in ['left', 'bottom', 'right', 'top']:
                                ax[0].spines[pos].set_edgecolor('k')
                                ax[0].spines[pos].set_edgecolor('k')

                        rise_dot_counter += 1
                    else:
                        pass

            for enu, chirp_time in enumerate(relevant_chirps):
                if frame_times[frames_oi[i]] - chirp_time > 0:
                    if chirp_passed[enu] == False:
                        chirp_dot, = ax[0].plot(0.05, 0.95, 'o', color='tab:orange', transform=ax[0].transAxes,
                                               markersize=20)
                        chirp_dot_handls.append(chirp_dot)
                        for pos in ['left', 'bottom', 'right', 'top']:
                            ax[0].spines[pos].set_edgecolor('tab:orange')
                            ax[0].spines[pos].set_edgecolor('tab:orange')
                        chirp_passed[enu] = True

                        chirp_active = True
                    if chirp_passed[enu] == True:
                        if chirp_dot_counter[enu] < 6:
                            chirp_dot_counter[enu] += 1
                        elif chirp_dot_counter[enu] == 6:
                            chirp_dot_handls[enu].remove()
                            for pos in ['left', 'bottom', 'right', 'top']:
                                ax[0].spines[pos].set_edgecolor('k')
                                ax[0].spines[pos].set_edgecolor('k')
                            chirp_dot_counter[enu] += 1
                        else:
                            pass

            label = (os.path.join(create_video_path,
                                  'frame%4.f.jpg' % len(glob.glob(os.path.join(create_video_path, '*.jpg'))))).replace(' ',
                                                                                                                       '0')
            plt.savefig(label, dpi=300)


    # for fish_nr in np.arange(2)[::-1]:
    #     for idx_oi in tqdm(np.array(rise_idx[fish_nr][~np.isnan(rise_idx[fish_nr])], dtype=int)):
    #         time_oi = times[idx_oi]
    #
    #         HH = int((time_oi / 3600) // 1)
    #         MM = int((time_oi - HH * 3600) // 60)
    #         SS =  int(time_oi - HH * 3600 - MM * 60)
    #
    #         frames_oi = np.arange(len(frame_times))[np.abs(frame_times - time_oi) <= dt]
    #         idxs_oi = np.arange(len(times))[np.abs(times - time_oi) <= dt*3]
    #
    #         fig = plt.figure(figsize=(16*2/2.54, 9*2/2.54))
    #         gs = gridspec.GridSpec(6, 2, left=0.075, bottom=0.05, right=1, top=0.95, width_ratios=(1.5, 3), hspace=.3, wspace=0.05)
    #         ax = []
    #         ax.append(fig.add_subplot(gs[:, 1]))
    #         ax.append(fig.add_subplot(gs[1:3, 0]))
    #         ax.append(fig.add_subplot(gs[3:5, 0]))
    #
    #
    #         y00, y01 = np.nanmin(fish_freqs[0][idxs_oi]), np.nanmax(fish_freqs[0][idxs_oi])
    #         y10, y11 = np.nanmin(fish_freqs[1][idxs_oi]), np.nanmax(fish_freqs[1][idxs_oi])
    #
    #         if y01 - y00 < 20:
    #             y01 = y00 + 20
    #         if y11 - y10 < 20:
    #             y11 = y10 + 20
    #         freq_span1 = (y01) - (y00)
    #         freq_span2 = (y11) - (y10)
    #
    #         yspan = freq_span1 if freq_span1 > freq_span2 else freq_span2
    #
    #         ax[1].plot(times[idxs_oi] - time_oi, fish_freqs[0][idxs_oi], marker='.', markersize=4, color='darkorange', lw=2, alpha=0.4)
    #         ax[2].plot(times[idxs_oi] - time_oi, fish_freqs[1][idxs_oi], marker='.', markersize=4,color='forestgreen', lw=2, alpha=0.4)
    #         ax[1].plot([0, 0], [y00 - yspan * 0.2, y00 + yspan * 1.3], '--', color='k')
    #         ax[2].plot([0, 0], [y10 - yspan * 0.2, y10 + yspan * 1.3], '--', color='k')
    #
    #         ax[1].set_xticks([-30, -15, 0, 15, 30])
    #         ax[2].set_xticks([-30, -15, 0, 15, 30])
    #         plt.setp(ax[1].get_xticklabels(), visible=False)
    #
    #         # spectrograms
    #         f_mask1 = np.arange(len(fill_freqs))[(fill_freqs >= y00 - yspan * 0.2) & (fill_freqs <= y00 + yspan * 1.3)]
    #         f_mask2 = np.arange(len(fill_freqs))[(fill_freqs >= y10 - yspan * 0.2) & (fill_freqs <= y10 + yspan * 1.3)]
    #         t_mask = np.arange(len(fill_times))[(fill_times >= time_oi-dt*4) & (fill_times <= time_oi+dt*4)]
    #
    #         ax[1].imshow(decibel(fill_spec[f_mask1[0]:f_mask1[-1], t_mask[0]:t_mask[-1]][::-1]),
    #                                           extent=[-dt*4, dt*4, y00 - yspan * 0.2, y00 + yspan * 1.3],
    #                                           aspect='auto',vmin = -100, vmax = -50, alpha=0.7, cmap='jet', interpolation='gaussian')
    #         ax[2].imshow(decibel(fill_spec[f_mask2[0]:f_mask2[-1], t_mask[0]:t_mask[-1]][::-1]),
    #                                           extent=[-dt*4, dt*4, y10 - yspan * 0.2, y10 + yspan * 1.3],
    #                                           aspect='auto',vmin = -100, vmax = -50, alpha=0.7, cmap='jet', interpolation='gaussian')
    #
    #         ax[1].set_ylim(y00 - yspan * 0.1, y00 + yspan * 1.2)
    #         ax[1].set_xlim(-dt*3, dt*3)
    #         ax[2].set_ylim(y10 - yspan * 0.1, y10 + yspan * 1.2)
    #         ax[2].set_xlim(-dt*3, dt*3)
    #
    #         ax[0].set_xticks([])
    #         ax[0].set_yticks([])
    #
    #         ax[1].tick_params(labelsize=12)
    #         ax[2].tick_params(labelsize=12)
    #
    #         ax[2].set_xlabel('time [s]', fontsize=14)
    #         fig.text(0.02, 0.5, 'frequency [Hz]', fontsize=14, va='center', rotation='vertical')
    #
    #         # plt.ion()
    #         for i in tqdm(np.arange(len(frames_oi))):
    #             # break
    #             video.set(cv2.CAP_PROP_POS_FRAMES, int(frames_oi[i]))
    #             ret, frame = video.read()
    #
    #             if i == 250:
    #                 dot, = ax[0].plot(0.05, 0.95, 'o', color='firebrick', transform = ax[0].transAxes, markersize=20)
    #             if i == 280:
    #                 dot.remove()
    #
    #             if i == 0:
    #                 img = ax[0].imshow(frame)
    #                 line1, = ax[1].plot([frame_times[frames_oi[i]] - time_oi, frame_times[frames_oi[i]] - time_oi],
    #                                    [y00 - yspan * 0.15, y00 + yspan * 1.3],
    #                                    color='k', lw=1)
    #                 line2, = ax[2].plot([frame_times[frames_oi[i]] - time_oi, frame_times[frames_oi[i]] - time_oi],
    #                                    [y10 - yspan * 0.15, y10 + yspan * 1.3],
    #                                    color='k', lw=1)
    #             else:
    #                 img.set_data(frame)
    #                 line1.set_data([frame_times[frames_oi[i]] - time_oi, frame_times[frames_oi[i]] - time_oi],
    #                               [y00 - yspan * 0.15, y00 + yspan * 1.3])
    #                 line2.set_data([frame_times[frames_oi[i]] - time_oi, frame_times[frames_oi[i]] - time_oi],
    #                               [y10 - yspan * 0.15, y10 + yspan * 1.3])
    #
    #             label = (os.path.join(create_video_path, 'frame%4.f.jpg' % len(glob.glob(os.path.join(create_video_path, '*.jpg'))))).replace(' ', '0')
    #             plt.savefig(label, dpi=300)
    #             # plt.pause(0.001)

            # quit()
        win_lose_str = 'lose'
        # video_name = ("./rise_video/%s_%2.f:%2.f:%2.f.mp4" % (win_lose_str, HH, MM, SS)).replace(' ', '0')
        # command = "ffmpeg -r 25 -i './rise_video/frame%4d.jpg' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -y -an"

        video_name = os.path.join(create_video_path, ("%s_%2.f:%2.f:%2.f.mp4" % (win_lose_str, HH, MM, SS)).replace(' ', '0'))
        command1 = "ffmpeg -r 25 -i"
        frames_path = '"%s"' % os.path.join(create_video_path, "frame%4d.jpg")
        command2 = "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -y -an"

        os.system(' '.join([command1, frames_path, command2, video_name]))
        os.system(' '.join(['rm', os.path.join(create_video_path, '*.jpg')]))
        # os.system(' '.join([command, video_name]))
        # os.system('rm ./rise_video/*.jpg')
        plt.close()
    embed()
    quit()


    ###############################
    fig, ax = plt.subplots()
    for i, c in enumerate(['firebrick', 'cornflowerblue']):
        ax.plot(times, fish_freqs[i], marker='.', color=c)
        r_idx = np.array(rise_idx[i][~np.isnan(rise_idx[i])], dtype=int)
        ax.plot(times[r_idx], fish_freqs[i][r_idx], 'o', color='k')
    pass
    ##############################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate videos around events.')
    parser.add_argument('file', type=str, help='folder/dataset to generate videos from.')
    parser.add_argument('-t', type=float, default=10, help='video duration before and after event.')
    # parser.add_argument("-c", action="store_true", help="check if LED pos is correct")
    # parser.add_argument('-x', type=int, nargs=2, default=[1272, 1282], help='x-borders of LED detect area (in pixels)')
    # parser.add_argument('-y', type=int, nargs=2, default=[1500, 1516], help='y-borders of LED area (in pixels)')
    args = parser.parse_args()
    main(args.file, args.t)