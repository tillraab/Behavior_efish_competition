import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import cv2
import glob
import argparse
from IPython import embed
from tqdm import tqdm

def main(folder, dt):
    video_path = glob.glob(os.path.join(folder, '2022*.mp4'))[0]
    create_video_path = os.path.join(folder, 'rise_video')
    if not os.path.exists(create_video_path):
        os.mkdir(create_video_path)
    video = cv2.VideoCapture(video_path) #  was 'cap'

    fish_freqs = np.load(os.path.join(folder, 'analysis', 'fish_freq_interp.npy'))
    max_freq, min_freq = np.nanmax(fish_freqs), np.nanmin(fish_freqs)
    rise_idx = np.load(os.path.join(folder, 'analysis', 'rise_idx.npy'))
    frame_times = np.load(os.path.join(folder, 'analysis', 'frame_times.npy'))
    times = np.load(os.path.join(folder, 'times.npy'))
    #######################################
    for fish_nr in np.arange(2)[::-1]:

        for idx_oi in tqdm(np.array(rise_idx[fish_nr][~np.isnan(rise_idx[fish_nr])], dtype=int)):
            # idx_oi = int(rise_idx[1][10])
            time_oi = times[idx_oi]

            # embed()
            # quit()

            HH = int((time_oi / 3600) // 1)
            MM = int((time_oi - HH * 3600) // 60)
            SS =  int(time_oi - HH * 3600 - MM * 60)

            frames_oi = np.arange(len(frame_times))[np.abs(frame_times - time_oi) <= dt]
            idxs_oi = np.arange(len(times))[np.abs(times - time_oi) <= dt*3]

            fig = plt.figure(figsize=(20/2.54, 20/2.54))
            gs = gridspec.GridSpec(2, 1, left=0.1, bottom = 0.1, right=0.95, top=0.95, height_ratios=(4, 1))
            ax = []
            ax.append(fig.add_subplot(gs[0, 0]))
            ax.append(fig.add_subplot(gs[1, 0]))
            ax[1].plot(times[idxs_oi] - time_oi, fish_freqs[0][idxs_oi], marker='.', color='firebrick')
            ax[1].plot(times[idxs_oi] - time_oi, fish_freqs[1][idxs_oi], marker='.', color='cornflowerblue')
            ax[1].set_ylim(min_freq - (max_freq-min_freq)*0.25, max_freq + (max_freq-min_freq)*0.25)
            ax[1].set_xlim(-dt*3, dt*3)
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            ax[1].tick_params(labelsize=12)
            ax[1].set_xlabel('time [s]', fontsize=14)
            # plt.ion()
            for i in tqdm(np.arange(len(frames_oi))):
                video.set(cv2.CAP_PROP_POS_FRAMES, int(frames_oi[i]))
                ret, frame = video.read()

                if i == 0:
                    img = ax[0].imshow(frame)
                    line, = ax[1].plot([frame_times[frames_oi[i]] - time_oi, frame_times[frames_oi[i]] - time_oi],
                                       [min_freq - (max_freq-min_freq)*0.25, max_freq + (max_freq-min_freq)*0.25],
                                       color='k', lw=1)
                else:
                    img.set_data(frame)
                    line.set_data([frame_times[frames_oi[i]] - time_oi, frame_times[frames_oi[i]] - time_oi],
                                  [min_freq - (max_freq-min_freq)*0.25, max_freq + (max_freq-min_freq)*0.25])

                # label = ('rise_video/frame%4.f.jpg' % len(glob.glob('rise_video/*.jpg'))).replace(' ', '0')
                label = (os.path.join(create_video_path, 'frame%4.f.jpg' % len(glob.glob(os.path.join(create_video_path, '*.jpg'))))).replace(' ', '0')
                plt.savefig(label, dpi=300)
                # plt.pause(0.001)

            win_lose_str = 'lose' if fish_nr == 1 else 'win'
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