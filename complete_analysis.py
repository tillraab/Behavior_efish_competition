import numpy as np
import pandas as pd
import os
import sys
from IPython import embed

def main(data_folder=None):
    trials_meta = pd.read_csv('order_meta.csv')

    for trial_idx in range(len(trials_meta)):
        group = trials_meta['group'][trial_idx]
        recording = trials_meta['recording'][trial_idx][1:-1]

        if group < 3:
            continue

        trial_path = os.path.join(data_folder, recording)
        if not os.path.exists(trial_path):
            continue

        if not os.path.exists(os.path.join(trial_path, 'led_idxs.csv')):
            continue

        print(group, recording)

        LED_on_idx_DF = pd.read_csv(os.path.join(trial_path, 'led_idxs.csv'))
        i0 = np.array([int(LED_on_idx_DF.keys()[0])])
        LED_on_idx_DATA = np.concatenate((i0, np.array(LED_on_idx_DF).T[0]))
        LED_on_time_BORIS = np.load(os.path.join(trial_path, 'LED_on_time.npy'), allow_pickle=True)

        print(len(LED_on_idx_DATA), len(LED_on_time_BORIS))

    embed()
    quit()
    pass

if __name__ == '__main__':
    main("/home/raab/data/mount_data/")