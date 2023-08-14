import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


from thunderfish.eventdetection import detect_peaks
from IPython import embed

class Trial(object):
    def __init__(self, folder, base_path, meta, fish_count):
        self._isValid = False

        self.base_path = base_path
        self.folder = folder

        self.meta = meta
        self.fish_count = fish_count

        self.light_sec = 3 * 60 * 60

        self.ids = None
        self.fish_freq = None
        self.fish_freq_interp = None
        self.fish_freq_val = None

        self.baseline_freq_times = None
        self.baseline_freqs = None

        self.rise_idxs = []
        self.rise_size = []

        self.fish_sign = None
        self.fish_sign_interp = None
        self.winner = None
        self.loser = None

        self.mean_shelter_power = None

        if os.path.exists(os.path.join(self.base_path, self.folder, 'fund_v.npy')):
            self.load()

    def __repr__(self):
        return f'Trial(Date={self.folder}, winner={self.winner})'
        # return self.folder

    def load(self):
        self.fund_v = np.load(os.path.join(self.base_path, self.folder, 'fund_v.npy'))
        self.idx_v = np.load(os.path.join(self.base_path, self.folder, 'idx_v.npy'))
        self.times = np.load(os.path.join(self.base_path, self.folder, 'times.npy'))
        self.ident_v = np.load(os.path.join(self.base_path, self.folder, 'ident_v.npy'))
        self.sign_v = np.load(os.path.join(self.base_path, self.folder, 'sign_v.npy'))

        self.ids = np.unique(self.ident_v[~np.isnan(self.ident_v)])
        if len(self.ids) == self.fish_count:
            self.isValid = True

    def reshape_and_interpolate(self):
        self.fish_freq = np.full((self.fish_count, len(self.times)), np.nan)
        self.fish_sign = np.full((self.fish_count, len(self.times), self.sign_v.shape[1]), np.nan)

        for enu, id in enumerate(self.ids):
            self.fish_freq[enu][self.idx_v[self.ident_v == id]] = self.fund_v[self.ident_v == id]
            self.fish_sign[enu][self.idx_v[self.ident_v == id]] = self.sign_v[self.ident_v == id]


        self.fish_freq_interp = np.full(self.fish_freq.shape, np.nan)
        self.fish_sign_interp = np.full(self.fish_sign.shape, np.nan)

        for enu, id in enumerate(self.ids):
            i0, i1 = self.idx_v[self.ident_v == id][0], self.idx_v[self.ident_v == id][-1]
            # self.fish_freq_interp[enu, i0:i1+1] = np.interp(self.times[i0:i1+1],
            #                                                 self.times[self.idx_v[self.ident_v == id]],
            #                                                 self.fish_freq[enu][~np.isnan(self.fish_freq[enu])])
            self.fish_freq_interp[enu, i0:i1+1] = np.interp(self.times[i0:i1+1],
                                                            self.times[self.idx_v[self.ident_v == id]],
                                                            self.fund_v[self.ident_v == id])

            # help_sign_v = list(map(lambda x: np.interp(self.times[i0:i1+1], self.times[self.idx_v[self.ident_v == id]], x),
            #                        self.fish_sign[enu][~np.isnan(self.fish_freq[enu])].T))
            help_sign_v = list(map(lambda x: np.interp(self.times[i0:i1+1], self.times[self.idx_v[self.ident_v == id]], x),
                                   self.sign_v[self.ident_v == id].T))
            self.fish_sign_interp[enu, i0:i1+1] = np.array(help_sign_v).T

    def baseline_freq(self, bw = 300):
        bins = np.arange(-bw / 2, self.times[-1] + bw / 2, bw)
        self.baseline_freq_times = np.array(bins[:-1] + (bins[1] - bins[0])/2)
        self.baseline_freqs = np.full((2, len(self.baseline_freq_times)), np.nan)
        self.pct95_freqs = np.full((2, len(self.baseline_freq_times)), np.nan)

        for enu, id in enumerate(self.ids):
            for i in range(len(bins) - 1):
                Cf = self.fish_freq[enu][(self.times > bins[i]) & (self.times <= bins[i + 1])]
                if len(Cf) == 0:
                    continue
                else:
                    self.baseline_freqs[enu][i] = np.nanpercentile(Cf, 5)
                    self.pct95_freqs[enu][i] = np.nanpercentile(Cf, 75)

        self.fish_freq_val = [np.nanmean(x[self.baseline_freq_times > self.light_sec]) for x in self.baseline_freqs]

    def winner_detection(self):
        day_mask = self.times > self.light_sec
        day_idxs = np.arange(len(self.times))[day_mask]

        shelter_power = np.empty((2, len(day_idxs)))
        for enu, id in enumerate(self.ids):
            shelter_power[enu] = self.fish_sign_interp[enu][day_idxs, -1]

        self.mean_shelter_power = np.nanmean(shelter_power, axis=1)
        self.winner = 1 if self.mean_shelter_power[1] > self.mean_shelter_power[0] else 0
        self.loser = 0 if self.winner == 1 else 1

    def rise_detection(self, rise_th):
        def check_rises_size(peak):
            peak_f = self.fish_freq[i][peak]
            peak_t = self.times[peak]

            closest_baseline_idx = list(map(lambda x: np.argmin(np.abs(self.baseline_freq_times - x)), peak_t))
            closest_baseline_freq = self.baseline_freqs[i][closest_baseline_idx]

            rise_size = peak_f - closest_baseline_freq

            return rise_size

        def correct_rise_idx(rise_peak_idx):

            rise_dt = np.diff(self.times[rise_peak_idx])
            rise_dt[rise_dt >= 10] = 10
            rise_dt[rise_dt < 10] = rise_dt[rise_dt < 10] - 1
            rise_dt = np.append(np.array([10]), rise_dt)


            freq_slope = np.full(np.shape(self.fish_freq)[1], np.nan)
            non_nan_idx = np.arange(len(freq_slope))[~np.isnan(self.fish_freq[i])]
            freq_slope[non_nan_idx[1:]] = np.diff(self.fish_freq[i][~np.isnan(self.fish_freq[i])])

            corrected_rise_idxs = []
            for enu, r_idx in enumerate(rise_peak_idx):
                mask = np.arange(len(freq_slope))[(self.times <= self.times[r_idx]) &
                                                  (self.times > self.times[r_idx] - rise_dt[enu]) &
                                                  (~np.isnan(freq_slope))]
                if len(mask) == 0:
                    corrected_rise_idxs.append(np.nan)
                else:
                    corrected_rise_idxs.append(mask[np.argmax(freq_slope[mask])])

            corrected_rise_idxs = np.array(corrected_rise_idxs)

            return corrected_rise_idxs

        for i in range(len(self.fish_freq)):
            rise_peak_idx, trough = detect_peaks(self.fish_freq[i][~np.isnan(self.fish_freq[i])], rise_th)
            non_nan_idx = np.arange(len(self.fish_freq[i]))[~np.isnan(self.fish_freq[i])]
            rise_peak_idx, trough = non_nan_idx[rise_peak_idx], non_nan_idx[trough]

            rise_size = check_rises_size(rise_peak_idx)

            rise_idx = correct_rise_idx(rise_peak_idx)
            # print(np.min(np.diff(self.times[rise_peak_idx])))

            self.rise_idxs.append(np.array(rise_idx[(rise_size >= rise_th) & (~np.isnan(rise_idx))], dtype=int))
            self.rise_size.append(rise_size[(rise_size >= rise_th) & (~np.isnan(rise_idx))])

    def update_meta(self):
        entries = self.meta.index.tolist()
        if self. folder not in entries:
            self.meta.loc[self.folder] = ['' for _ in self.meta.columns]
        self.meta.loc[self.folder, 'Win_ID'] = self.ids[self.winner]
        self.meta.loc[self.folder, 'Lose_ID'] = self.ids[self.loser]

        self.meta.loc[self.folder, 'Win_EODf'] = self.fish_freq_val[self.winner]
        self.meta.loc[self.folder, 'Lose_EODf'] = self.fish_freq_val[self.loser]

        self.meta.loc[self.folder, 'Win_rise_c'] = len(self.rise_idxs[self.winner])
        self.meta.loc[self.folder, 'Lose_rise_c'] = len(self.rise_idxs[self.loser])

        self.meta.loc[self.folder, 'light_sec'] = self.light_sec

        self.meta.to_csv(os.path.join(self.base_path, 'meta.csv'), sep =',')

    def ilustrate(self):
        fig = plt.figure(figsize=(20/2.54, 12/2.54))
        gs = gridspec.GridSpec(1, 1, left = 0.1, bottom = 0.1, right = 0.95, top = 0.95)
        ax = fig.add_subplot(gs[0, 0])

        for enu, id in enumerate(self.ids):
            c = 'firebrick' if self.winner == enu else 'forestgreen'
            ax.plot(self.times/3600, self.fish_freq[enu], marker='.', color=c, zorder=1, label=f'{self.mean_shelter_power[enu]:.2f}dB')
            ax.plot(self.times[np.isnan(self.fish_freq[enu])]/3600, self.fish_freq_interp[enu][np.isnan(self.fish_freq[enu])], '.', zorder=1, color=c, alpha=0.25)
            ax.plot(self.baseline_freq_times/3600, self.baseline_freqs[enu], '--', color='k', zorder=2)
            ax.plot(self.baseline_freq_times/3600, self.pct95_freqs[enu], '--', color='k', zorder=2)

            ax.plot(self.times[self.rise_idxs[enu]]/3600, self.fish_freq_interp[enu][self.rise_idxs[enu]], 'o', color='k')


            win_str = '(W)' if self.winner == enu else ''

            ax.text(self.times[-1]/3600, self.fish_freq_val[enu]-10, '%.0f' % id + win_str, va ='center', ha='right')

            ax.set_xlim(0, self.times[-1]/3600)

            freq_range = (np.nanmin(self.fish_freq), np.nanmax(self.fish_freq))
            ax.set_ylim(freq_range[0] - 20, freq_range[1] + 10)
        ax.legend(loc = 'upper right', bbox_to_anchor=(1, 1))
        ax.set_title(self.folder)
        plt.show()

    def save(self):
        saveorder = -1 if self.winner == 1 else 1

        if not os.path.exists(os.path.join(self.base_path, self.folder, 'analysis')):
            os.mkdir(os.path.join(self.base_path, self.folder, 'analysis'))

        np.save(os.path.join(self.base_path, self.folder, 'analysis', 'ids.npy'), self.ids[::saveorder])

        np.save(os.path.join(self.base_path, self.folder, 'analysis', 'fish_freq.npy'), self.fish_freq[::saveorder])
        np.save(os.path.join(self.base_path, self.folder, 'analysis', 'fish_freq_interp.npy'), self.fish_freq_interp[::saveorder])

        np.save(os.path.join(self.base_path, self.folder, 'analysis', 'baseline_freqs.npy'), self.baseline_freqs[::saveorder])
        np.save(os.path.join(self.base_path, self.folder, 'analysis', 'baseline_freq_times.npy'), self.baseline_freq_times[::saveorder])

        help_lens = [len(x) for x in self.rise_idxs]
        rise_idxs_s = np.full((self.fish_count, np.max(help_lens)), np.nan)
        rise_size_s = np.full((self.fish_count, np.max(help_lens)), np.nan)
        for i in range(self.fish_count):
            rise_idxs_s[i][:len(self.rise_idxs[i])] = self.rise_idxs[i]
            rise_size_s[i][:len(self.rise_size[i])] = self.rise_size[i]
        np.save(os.path.join(self.base_path, self.folder, 'analysis', 'rise_idx.npy'), rise_idxs_s[::saveorder])
        np.save(os.path.join(self.base_path, self.folder, 'analysis', 'rise_size.npy'), rise_size_s[::saveorder])

    @property
    def isValid(self):
        return self._isValid

    @isValid.setter
    def isValid(self, value):
        print('Trial (%s) is valid' % (self.folder))
        self._isValid = value

    def frame_to_idx(self, event_frames):
        self.sr = 20000
        LED_idx = pd.read_csv(os.path.join(self.folder, 'led_idxs.csv'), sep=',', encoding = "utf-7")

        led_idx = np.array(LED_idx).T[0]
        led_frame = np.load(os.path.join(self.folder, 'LED_frames.npy'))

        led_idx_span = led_idx[-1] - led_idx[0]
        led_frame_span = led_frame[-1] - led_frame[0]

        frames_to_idx = ((event_frames - led_frame[0]) / led_frame_span) * led_idx_span + led_idx[0]

        event_times = frames_to_idx / self.sr

        return event_times


def main():
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('file', type=str, help='single recording analysis', default='')
    parser.add_argument('-d', "--dev", action="store_true", help="developer mode; no data saved")
    # parser.add_argument('-x', type=int, nargs=2, default=[1272, 1282], help='x-borders of LED detect area (in pixels)')
    # parser.add_argument('-y', type=int, nargs=2, default=[1500, 1516], help='y-borders of LED area (in pixels)')
    args = parser.parse_args()

    base_path = None
    folders = []
    for root, dirs, files in os.walk(args.file):
        for file in files:
            if file.endswith('.raw'):
                root = os.path.normpath(root)
                print(root, file)
                print(os.path.join(root, file))
                folders.append(os.path.split(root)[-1])
                if not base_path:
                    base_path = os.path.split(root)[0]
    folders = sorted(folders)

    if os.path.exists(os.path.join(base_path, 'meta.csv')) and not args.dev:
        meta = pd.read_csv(os.path.join(base_path, 'meta.csv'), sep=',', index_col=0, encoding = "utf-7")
    else:
        meta = None

    # embed()
    # if args.f == '':
    #     folders = os.listdir(args.f)
    #     folders = [x for x in folders if not '.' in x]
    # else:
    #     folders= [os.path.split(os.path.normpath(args.f))[-1]]
    # folders = sorted(folders)

    trials = []
    for folder in folders:
        trial = Trial(folder, base_path, meta, fish_count=2)
        if not trial.isValid:
            continue

        trial.reshape_and_interpolate()
        trial.winner_detection()
        trial.baseline_freq(bw=300)

        # ToDo: q10 corrected EODfs

        trial.rise_detection(rise_th=5)

        if meta is not None:
            if not args.dev:
                trial.update_meta()
        if not args.dev:
            trial.save()
        trial.ilustrate()
        trials.append(trial)

        # meta.loc[folder, 'Fish1_ID'] = 1
        # meta.to_csv('')

if __name__ == '__main__':
    main()