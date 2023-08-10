import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import scipy.stats as scp
import networkx as nx

from thunderfish.powerspectrum import decibel

from IPython import embed
from event_time_correlations import load_and_converete_boris_events
glob_colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'k']


def plot_transition_matrix(matrix, labels):
    fig = plt.figure(figsize=(20/2.54, 20/2.54))
    #gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.1, width_ratios=[8, 1])
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.925, top=0.95)
    ax = fig.add_subplot(gs[0, 0])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # cax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(matrix)
    ax.set_xticks(list(range(len(matrix))))
    ax.set_yticks(list(range(len(matrix))))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.tick_params(labelsize=10)
    cax.tick_params(labelsize=10)
    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'markov', 'event_counts' + '.png'), dpi=300)
    plt.close()


def plot_transition_diagram(matrix, labels, node_size, ax, threshold=5,
                            color_by_origin=False, color_by_target=False, title=''):



    matrix[matrix <= threshold] = 0
    matrix = np.around(matrix, decimals=1)
    Graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

    node_labels = dict(zip(Graph, labels))
    # Graph = nx.relabel_nodes(Graph, node_labels)

    edge_labels = nx.get_edge_attributes(Graph, 'weight')
    positions = nx.circular_layout(Graph)
    positions2 = nx.circular_layout(Graph)
    for p in positions:
        positions2[p][0] *= 1.2
        positions2[p][1] *= 1.2

    # ToDo: nodes
    nx.draw_networkx_nodes(Graph, pos=positions, node_size=node_size, ax=ax, alpha=0.5, node_color=np.array(glob_colors)[:len(node_size)])
    nx.draw_networkx_labels(Graph, pos=positions2, labels=node_labels, ax=ax)
    # google networkx drawing to get better graphs with networkx
    # nx.draw(Graph, pos=positions, node_size=node_size, label=labels, with_labels=True, ax=ax)
    # # ToDo: edges
    edge_width = np.array([x / 5 for x in [*edge_labels.values()]])
    if color_by_origin:
        edge_colors = np.array(glob_colors)[np.array([*edge_labels.keys()], dtype=int)[:, 0]]
    elif color_by_target:
        edge_colors = np.array(glob_colors)[np.array([*edge_labels.keys()], dtype=int)[:, 1]]
    else:
        edge_colors = 'k'


    edge_width[edge_width >= 6] = 6

    nx.draw_networkx_edges(Graph, pos=positions, node_size=node_size, width=edge_width,
                           arrows=True, arrowsize=20,
                           min_target_margin=25, min_source_margin=25, connectionstyle="arc3, rad=0.025",
                           ax=ax, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(Graph, positions, label_pos=0.2, edge_labels=edge_labels, ax=ax, rotate=True)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title(title, fontsize=12)

def create_marcov_matrix(individual_event_times, individual_event_labels):
    event_times = []
    event_labels = []
    for ll, t in zip(individual_event_labels, individual_event_times):
        event_times.extend(t)
        event_labels.extend(np.full(len(t), ll))

    time_sorter = np.argsort(event_times)
    event_times = np.array(event_times)[time_sorter]
    event_labels = np.array(event_labels)[time_sorter]

    marcov_matrix = np.zeros((len(individual_event_labels) + 1, len(individual_event_labels) + 1))
    for enu_ori, label_ori in enumerate(individual_event_labels):
        for enu_tar, label_tar in enumerate(individual_event_labels):
            n = len(event_times[:-1][(event_labels[:-1] == label_ori) & (event_labels[1:] == label_tar) & (
                        np.diff(event_times) <= 5)])
            marcov_matrix[enu_ori, enu_tar] = n
    for enu_tar, label_tar in enumerate(individual_event_labels):
        n = len(event_times[:-1][(event_labels[1:] == label_tar) & (np.diff(event_times) > 5)])
        marcov_matrix[-1, enu_tar] = n
    marcov_matrix[-1, 5] = 0

    individual_event_labels.append('void')

    ### get those cases where ag_on does not point to event and no event points to corresponding ag_off ... add thise cases in marcov matrix
    chase_on_idx = np.where(event_labels == individual_event_labels[4])[0]
    chase_off_idx = np.where(event_labels == individual_event_labels[5])[0]
    helper_mask = np.ones_like(chase_on_idx)
    helper_mask[np.diff(event_times)[chase_on_idx] <= 5] = 0
    helper_mask[np.diff(event_times)[chase_off_idx - 1] <= 5] = 0

    marcov_matrix[4, 5] += np.sum(helper_mask)

    return marcov_matrix

def fine_spec_plot(ax, example_1_path, trial_summary, example_ag_on_off):

    ex1_df_idx = trial_summary[trial_summary['recording'] == os.path.split(example_1_path)[-1]].index.to_numpy()[0]
    lose_id = trial_summary.iloc[ex1_df_idx]['lose_ID']

    fine_spec_shape = np.load(os.path.join(example_1_path, 'fill_spec_shape.npy'))
    fine_spec = np.memmap(os.path.join(example_1_path, 'fill_spec.npy'), dtype='float', mode='r', shape=(fine_spec_shape[0], fine_spec_shape[1]), order='F')
    fine_times = np.load(os.path.join(example_1_path, 'fill_times.npy'))
    spec_freqs = np.load(os.path.join(example_1_path, 'fill_freqs.npy'))

    times = np.load(os.path.join(example_1_path, 'times.npy'))
    fund_v = np.load(os.path.join(example_1_path, 'fund_v.npy'))
    ident_v = np.load(os.path.join(example_1_path, 'ident_v.npy'))
    idx_v = np.load(os.path.join(example_1_path, 'idx_v.npy'))

    # artificial_t_axis = np.linspace(times[0], times[-1], spec.shape[1])
    # artificial_f_axis = np.linspace(0, 2000, spec.shape[0])
    # plt.pcolormesh(artificial_t_axis, artificial_f_axis, decibel(spec), vmin=-100, vmax=-50)
    lose_freq_in_snippet = fund_v[(ident_v == lose_id) & (times[idx_v] > example_ag_on_off[0][0]-5) & (times[idx_v] < example_ag_on_off[0][1]+5)]
    max_f, min_f = np.max(lose_freq_in_snippet) + 25, np.min(lose_freq_in_snippet) - 25

    f_idx0 = np.where(spec_freqs <= min_f)[0][-1]
    f_idx1 = np.where(spec_freqs >= max_f)[0][0]

    t_idx0 = np.where(fine_times <= example_ag_on_off[0][0] - 5)[0][-1]
    t_idx1 = np.where(fine_times >= example_ag_on_off[0][0] + 4)[0][0]
    ax.pcolormesh(fine_times[t_idx0:t_idx1+1] - example_ag_on_off[0][0], spec_freqs[f_idx0:f_idx1+1],
                  decibel(fine_spec[f_idx0:f_idx1+1, t_idx0:t_idx1+1]))

    t_idx0 = np.where(fine_times <= example_ag_on_off[0][1] - 5)[0][-1]
    t_idx1 = np.where(fine_times >= example_ag_on_off[0][1] + 5)[0][0]
    ax.pcolormesh(fine_times[t_idx0:t_idx1+1] - example_ag_on_off[0][1] + 10, spec_freqs[f_idx0:f_idx1+1],
                  decibel(fine_spec[f_idx0:f_idx1+1, t_idx0:t_idx1+1]))

    ax.fill_between([4, 5], [spec_freqs[f_idx0], spec_freqs[f_idx0]], [spec_freqs[f_idx1], spec_freqs[f_idx1]], color='white')

def main(base_path):
    if not os.path.exists(os.path.join(os.path.split(__file__)[0], 'figures', 'markov')):
        os.makedirs(os.path.join(os.path.split(__file__)[0], 'figures', 'markov'))

    trial_summary = pd.read_csv(os.path.join(base_path, 'trial_summary.csv'), index_col=0)
    chirp_notes = pd.read_csv(os.path.join(base_path, 'chirp_notes.csv'), index_col=0)
    # trial_summary = trial_summary[chirp_notes['good'] == 1]
    trial_mask = chirp_notes['good'] == 1

    all_marcov_matrix = []
    all_event_counts = []
    all_agonistic_categorie = []

    # agonistic categorie plot
    # fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
    # gs = gridspec.GridSpec(2, 1, left=0.1, bottom=0.1, right=0.9, top=0.95, height_ratios=[1, 4], hspace=0)
    # ax = fig.add_subplot(gs[1, 0])
    # ax_spec = fig.add_subplot(gs[0, 0], sharex=ax)
    # plt.setp(ax_spec.get_xticklabels(), visible=False)
    #
    # for i in range(1, 5):
    #     ax.fill_between([0, 4], np.array([-.2, -.2]) + i, np.array([.2, .2]) + i, color='tab:grey')
    #     ax.fill_between([5, 10], np.array([-.2, -.2]) + i, np.array([.2, .2]) + i, color='tab:grey')
    #
    #     fill_dots = np.arange(4, 5.1, 0.125)
    #     ax.plot(fill_dots, np.ones_like(fill_dots)*i, '.', color='tab:grey', markersize=3)

    got_examples = [False, False, False]
    example_ag_on_off = [[], [], []]
    example_chirp_times = [[], [], []]
    example_rise_times = [[], [], []]
    example_1_path = ''
    example_skips = [15, 4, 3] #3, 5, 9, 15, 19

    for index, trial in trial_summary.iterrows():
        trial_path = os.path.join(base_path, trial['recording'])

        if not trial_mask[index]:
            continue
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

        # trial marcov matrix
        individual_event_times = [chirp_times[1], rise_times[1], chirp_times[0], rise_times[0], ag_on_off_t_GRID[:, 0],
                                  ag_on_off_t_GRID[:, 1], contact_t_GRID]
        individual_event_labels = [r'chirp$_{lose}$', r'rise$_{lose}$', r'chirp$_{win}$', r'rise$_{win}$',
                                   r'chace$_{on}$', r'chace$_{off}$', 'contact']

        marcov_matrix = create_marcov_matrix(individual_event_times, individual_event_labels)
        all_marcov_matrix.append(marcov_matrix)

        # compute and store trial event counts
        event_counts = np.array(list(map(lambda x: len(x), individual_event_times)))
        event_counts = np.append(event_counts, marcov_matrix[-1].sum())
        all_event_counts.append(event_counts)

        # agonistic categories
        agonitic_categorie = np.zeros(len(ag_on_off_t_GRID))
        for enu, (chase_on_time, chase_off_time) in enumerate(ag_on_off_t_GRID):
            chase_dur = chase_off_time - chase_on_time
            chirp_dt = chase_dur if chase_dur < 5 else 5
            max_dt = 5

            # check if rise before chase / chirp at end
            rise_before, chirp_arround_end = False, False
            if np.any(((chase_on_time - rise_times[1]) > 0) & ((chase_on_time - rise_times[1]) < max_dt)):
                rise_times_oi = rise_times[1][((chase_on_time - rise_times[1]) > 0) & ((chase_on_time - rise_times[1]) < max_dt)]
                rise_before = True

            if np.any( ((chase_off_time - chirp_times[1]) < chirp_dt) & ((chirp_times[1] - chase_off_time) < max_dt)):

                chirp_time_oi = chirp_times[1][((chase_off_time - chirp_times[1]) < chase_dur) & ((chirp_times[1] - chase_off_time) < max_dt)]
                chirp_arround_end = True


            # define agonistic categorie based on rise/chirp occurance
            if rise_before:
                if chirp_arround_end:
                    agonitic_categorie[enu] = 1
                else:
                    agonitic_categorie[enu] = 2
            else:
                if chirp_arround_end:
                    agonitic_categorie[enu] = 3
                else:
                    agonitic_categorie[enu] = 4

            if agonitic_categorie[enu] == 1 and not got_examples[0]:
                if chase_dur > 10:
                    if np.any((chirp_time_oi - chase_off_time) < 0) and np.any((chirp_time_oi - chase_off_time) > 0):
                        if example_skips[int(agonitic_categorie[enu] - 1)] == 0:
                            example_ag_on_off[int(agonitic_categorie[enu] - 1)].extend([chase_on_time, chase_off_time])
                            example_chirp_times[int(agonitic_categorie[enu] - 1)].extend(chirp_time_oi)
                            example_rise_times[int(agonitic_categorie[enu] - 1)].extend(rise_times_oi)
                            example_1_path = trial_path
                            got_examples[0] = True
                        else:
                            example_skips[int(agonitic_categorie[enu] - 1)] -= 1
            elif agonitic_categorie[enu] == 2 and not got_examples[1]:
                if chase_dur > 10:
                    if example_skips[int(agonitic_categorie[enu] - 1)] == 0:
                        example_ag_on_off[int(agonitic_categorie[enu] - 1)].extend([chase_on_time, chase_off_time])
                        example_rise_times[int(agonitic_categorie[enu] - 1)].extend(rise_times_oi)
                        got_examples[1] = True
                    else:
                        example_skips[int(agonitic_categorie[enu] - 1)] -= 1
            elif agonitic_categorie[enu] == 3 and not got_examples[2]:
                if chase_dur > 10:
                    if np.any((chirp_time_oi - chase_off_time) < 0) and np.any((chirp_time_oi - chase_off_time) > 0):
                        if example_skips[int(agonitic_categorie[enu] - 1)] == 0:
                            example_ag_on_off[int(agonitic_categorie[enu] - 1)].extend([chase_on_time, chase_off_time])
                            example_chirp_times[int(agonitic_categorie[enu] - 1)].extend(chirp_time_oi)
                            got_examples[2] = True
                        else:
                            example_skips[int(agonitic_categorie[enu] - 1)] -= 1
            else:
                pass

        all_agonistic_categorie.append(agonitic_categorie)

    ###  agonistic categorie example figure
    fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
    gs = gridspec.GridSpec(2, 1, left=0.1, bottom=0.1, right=0.9, top=0.9, height_ratios=[1, 4], hspace=0)
    ax = fig.add_subplot(gs[1, 0])
    ax_spec = fig.add_subplot(gs[0, 0], sharex=ax)
    plt.setp(ax_spec.get_xticklabels(), visible=False)

    for i in range(1, 5):
        ax.fill_between([0, 4], np.array([-.2, -.2]) + i, np.array([.2, .2]) + i, color='tab:grey')
        ax.fill_between([5, 10], np.array([-.2, -.2]) + i, np.array([.2, .2]) + i, color='tab:grey')

        fill_dots = np.arange(4, 5.1, 0.125)
        ax.plot(fill_dots, np.ones_like(fill_dots)*i, '.', color='tab:grey', markersize=3)

    for enu, (chirp_time_oi, rise_times_oi, ag_on_off) in enumerate(zip(example_chirp_times, example_rise_times, example_ag_on_off)):
        chase_on_time, chase_off_time = ag_on_off

        for ct in chirp_time_oi:
            ax.plot([ct - chase_off_time + 10, ct - chase_off_time + 10], [enu + .8, enu + 1.2], color='k', lw=2)
        for rt in rise_times_oi:
            ax.plot([rt - chase_on_time, rt - chase_on_time], [enu + .8, enu + 1.2], color='firebrick', lw=2)

    stacked_agonistic_categories = np.hstack(all_agonistic_categorie)
    pct_each_categorie = np.zeros(4)
    for enu, cat in enumerate(range(1, 5)):
        pct_each_categorie[enu] = len(stacked_agonistic_categories[stacked_agonistic_categories == cat]) / len(stacked_agonistic_categories)
        ax.text(15.2, enu + 1, f'{pct_each_categorie[enu] * 100:.1f}' + ' $\%$', clip_on=False, fontsize=14, ha='left', va='center')

    # plot correct spectrogram
    fine_spec_plot(ax_spec, example_1_path, trial_summary, example_ag_on_off)

    ##########################################

    ax.plot([0, 0], [0.5, 5], '--', color='k', lw=1)
    ax.plot([10, 10], [0.5, 5], '--', color='k', lw=1)
    ax.set_ylim(0.5, 4.5)
    ax.set_xlim(-5, 15)
    ax.set_yticks([1, 2, 3, 4])
    # ax.set_yticklabels([r'rise$_{pre}$  $&$ chirp$_{end}$', r'only rise$_{pre}$', r'only chirp$_{end}$', 'no communication'])
    ax.set_yticklabels(['A  ', 'B  ', 'C  ', 'D  '])
    ax.invert_yaxis()
    ax.set_xlabel('time [s]', fontsize=12)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis = 'x', labelsize=10)


    legend_elements = [Line2D([0], [0], color='firebrick', lw=2, label=r'rise$_{lose}$'),
                       Line2D([0], [0], color='k', lw=2, label=r'chirp$_{lose}$'),
                       Patch(facecolor='tab:grey', edgecolor='w', label= 'chase event')]

    ax_spec.legend(handles=legend_elements, loc='lower right', ncol=3, bbox_to_anchor=(1, 1), frameon=False, fontsize=10, facecolor='white')
    ax_spec.set_ylabel('EODf [Hz]', fontsize=12)
    ax.spines[['right', 'top']].set_visible(False)

    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'markov', 'agonistic_categories' + '.png'), dpi=300)
    plt.show()


    ### bar plot - agonistic categories counts/pct #####################################################################

    fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54))
    ax.bar(np.arange(4),
           [len(stacked_agonistic_categories[stacked_agonistic_categories == 1]),
            len(stacked_agonistic_categories[stacked_agonistic_categories == 2]),
            len(stacked_agonistic_categories[stacked_agonistic_categories == 3]),
            len(stacked_agonistic_categories[stacked_agonistic_categories == 4])])
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([r'rise$_{pre}$ + chirp$_{end}$', r'rise$_{pre}$ + _', r'_ + chirp$_{end}$', '_ + _'])
    plt.show()

    # pct
    pct_agon_categorie = np.zeros(shape=(len(all_agonistic_categorie), 4))
    for enu, agonitic_categorie in enumerate(all_agonistic_categorie):
        for cat in np.arange(4):
            pct_agon_categorie[enu, cat] = len(agonitic_categorie[agonitic_categorie == cat+1]) / len(agonitic_categorie)

    fig, ax = plt.subplots(figsize=(20 / 2.54, 12 / 2.54))
    ax.bar(np.arange(4), pct_agon_categorie.mean(0))
    ax.errorbar(np.arange(4), pct_agon_categorie.mean(0), yerr=pct_agon_categorie.std(0), fmt='', color='k', linestyle='None')
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([r'rise$_{pre}$ + chirp$_{end}$', r'rise$_{pre}$ + _', r'_ + chirp$_{end}$', '_ + _'])
    plt.show()


    ### marcov models plots ############################################################################################
    all_marcov_matrix = np.array(all_marcov_matrix)
    all_event_counts = np.array(all_event_counts)

    collective_marcov_matrix = np.sum(all_marcov_matrix, axis=0)
    collective_event_counts = np.sum(all_event_counts, axis=0)

    plot_transition_matrix(collective_marcov_matrix, individual_event_labels)

    fig, ax = plt.subplots(figsize=(21 / 2.54, 19 / 2.54))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

    plot_transition_diagram(
        collective_marcov_matrix / collective_event_counts.reshape(len(collective_event_counts), 1) * 100,
        individual_event_labels, collective_event_counts, ax, threshold=5, color_by_origin=True, title='origin triggers target [%]')
    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'markov', 'markov_destination' + '.png'), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(21 / 2.54, 19 / 2.54))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plot_transition_diagram(collective_marcov_matrix / collective_event_counts * 100,
                            individual_event_labels, collective_event_counts, ax, threshold=5, color_by_target=True,
                            title='target triggered by origin [%]')
    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'markov', 'markov_origin' + '.png'), dpi=300)
    plt.close()

    for i, (marcov_matrix, event_counts) in enumerate(zip(all_marcov_matrix, all_event_counts)):
        fig, ax = plt.subplots(figsize=(21 / 2.54, 19 / 2.54))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

        plot_transition_diagram(
            marcov_matrix / event_counts.reshape(len(event_counts), 1) * 100,
            individual_event_labels, event_counts, ax, threshold=5, color_by_origin=True,
            title='origin triggers target [%]')
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'markov', f'markov_{i}_destination' + '.png'),
                    dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(21 / 2.54, 19 / 2.54))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        plot_transition_diagram(marcov_matrix / event_counts * 100,
                                individual_event_labels, event_counts, ax, threshold=5, color_by_target=True,
                                title='target triggered by origin [%]')
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', 'markov', f'markov_{i}_origin' + '.png'),
                    dpi=300)
        plt.close()
    ####################################################################################################################

    embed()
    quit()
    pass


if __name__ == '__main__':
    main(sys.argv[1])
