import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def check_LED(cap, frame_count, x0, x1, y0, y1):
    fig, ax = plt.subplots()
    ax.plot([x0, x0], [y0, y1], 'r')
    ax.plot([x1, x1], [y0, y1], 'r')
    ax.plot([x0, x1], [y0, y0], 'r')
    ax.plot([x0, x1], [y1, y1], 'r')
    plt.ion()

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count / 2))
    f = None
    try:
        for i in np.arange(int(frame_count / 2), frame_count):
            ret, frame = cap.read()
            if f == None:
                f = ax.imshow(frame)
            else:
                f.set_data(frame)

            sum_frame = np.sum(frame, axis=2)
            LED_v = np.mean(sum_frame[y0:y1, x0:x1])
            print('%.0f: %.1f \n' % (i, LED_v))
            plt.pause(0.001)
    except KeyboardInterrupt:
        plt.close()
        quit()
    quit()

def main(file_path, check, x, y):
    folder, filename = os.path.split(file_path)
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    x0, x1 = x
    y0, y1 = y

    if check:
        check_LED(cap, frame_count, x0, x1, y0, y1)

    ###############
    # cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count / 2))
    # frame_count = 1000
    ########################

    light_th = 100
    LED_val = np.zeros(frame_count)
    print('Frame_count: %.0f' % frame_count)
    for i in range(frame_count):
        if i % 1000 == 0:
            print('progress: %.1f' % ((i/frame_count)*100) + '%')
        ret, frame = cap.read()

        sum_frame = np.sum(frame, axis=2)
        LED_val[i] = np.mean(sum_frame[y0:y1, x0:x1])

    np.save(os.path.join(folder, 'LED_val.npy'), LED_val)

    LED_frames = np.arange(len(LED_val)-1)[(LED_val[:-1] < light_th) & (LED_val[1:] > light_th)]

    np.save(os.path.join(folder, 'LED_frames.npy'), LED_frames)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(LED_val)), LED_val, color='k')
    ax.plot(LED_frames, np.ones(len(LED_frames))*light_th, 'o', color='firebrick')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect frames of blinking LED in video recordings.')
    parser.add_argument('file', type=str, help='video file to be analyzed')
    parser.add_argument("-c", '--check', action="store_true", help="check if LED pos is correct")
    parser.add_argument('-x', type=int, nargs=2, default=[1240, 1250], help='x-borders of LED detect area (in pixels)')
    parser.add_argument('-y', type=int, nargs=2, default=[1504, 1526], help='y-borders of LED area (in pixels)')
    args = parser.parse_args()
    import glob

    main(args.file, args.check, args.x, args.y)
