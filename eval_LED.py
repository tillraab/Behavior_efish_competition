import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from IPython import embed
import cv2
import glob

def main(folder):
    sr = 20000

    video_path = glob.glob(os.path.join(folder, '2022*.mp4'))[0]
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    times = np.load(os.path.join(folder, 'times.npy'))
    LED_idx = pd.read_csv(os.path.join(folder, 'led_idxs.csv'), sep=',')

    led_idx = np.array(LED_idx).T[0]
    led_frame = np.load(os.path.join(folder, 'LED_frames.npy'))
    led_vals = np.load(os.path.join(folder, 'LED_val.npy'))

    led_idx_span = led_idx[-1] - led_idx[0]
    led_frame_span = led_frame[-1] - led_frame[0]

    led_frame_to_idx = ((led_frame-led_frame[0]) / led_frame_span) * led_idx_span + led_idx[0]

    frame_idxs = np.arange(frame_count)
    frame_times = (((frame_idxs - led_frame[0]) / led_frame_span) * led_idx_span + led_idx[0]) / sr

    if not os.path.exists(os.path.join(folder, 'analysis')):
        os.mkdir(os.path.join(folder, 'analysis'))
    np.save(os.path.join(folder, 'analysis', 'frame_times.npy'), frame_times)

    ########################################################################################
    fig, ax = plt.subplots()
    ax.plot(led_vals)
    ax.plot(led_frame, np.ones(len(led_frame))*100, '.', color='firebrick')

    ########################################################################################
    fig, ax = plt.subplots()
    ax.plot(led_idx / sr, np.ones(len(led_idx)), '.', color='k')
    ax.plot(led_frame_to_idx / sr, np.ones(len(led_frame_to_idx))+.1, '.', color='firebrick')
    ax.plot([times[0], times[0]], [0.5, 1.5], 'k', lw=1)
    ax.plot([times[-1], times[-1]], [0.5, 1.5], 'k', lw=1)

    ax.plot(frame_times, np.ones(len(frame_times))*0.5)

    ax.set_ylim(0, 2)

    plt.show()

    embed()
    quit()
    pass

if __name__ == '__main__':
    main(sys.argv[1])