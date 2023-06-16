import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.stats as scp
import networkx as nx

from IPython import embed
from event_time_correlations import load_and_converete_boris_events
glob_colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']


def plot_transition_matrix(matrix, labels):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    ax.set_xticks(list(range(len(matrix))))
    ax.set_yticks(list(range(len(matrix))))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    fig.colorbar(im)
    plt.show()


def plot_transition_diagram(matrix, labels, node_size, threshold=5, save_str = 'rdm'):
    matrix[matrix <= threshold] = 0
    matrix = np.around(matrix, decimals=1)
    fig, ax = plt.subplots(figsize=(21 / 2.54, 19 / 2.54))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
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
    nx.draw_networkx_labels(Graph, pos=positions2, labels=node_labels)
    # google networkx drawing to get better graphs with networkx
    # nx.draw(Graph, pos=positions, node_size=node_size, label=labels, with_labels=True, ax=ax)
    # # ToDo: edges
    edge_width = np.array([x / 5 for x in [*edge_labels.values()]])
    edge_colors = np.array(glob_colors)[np.array([*edge_labels.keys()], dtype=int)[:, 0]]


    edge_width[edge_width >= 6] = 6
    # nx.draw_networkx_edges(Graph, pos=positions, node_size=node_size, width=edge_width,
    #                        arrows=True, arrowsize=20,
    #                        min_target_margin=25, min_source_margin=10, connectionstyle="arc3, rad=0.0",
    #                        ax=ax)
    # nx.draw_networkx_edge_labels(Graph, positions, label_pos=0.5, edge_labels=edge_labels, ax=ax, rotate=False)

    nx.draw_networkx_edges(Graph, pos=positions, node_size=node_size, width=edge_width,
                           arrows=True, arrowsize=20,
                           min_target_margin=25, min_source_margin=10, connectionstyle="arc3, rad=0.025",
                           ax=ax, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(Graph, positions, label_pos=0.2, edge_labels=edge_labels, ax=ax, rotate=True)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    # plt.title(title)
    plt.savefig(os.path.join(os.path.split(__file__)[0], 'figures', save_str + '.png'), dpi=300)
    plt.close()

def main(base_path):
    trial_summary = pd.read_csv(os.path.join(base_path, 'trial_summary.csv'), index_col=0)
    chirp_notes = pd.read_csv(os.path.join(base_path, 'chirp_notes.csv'), index_col=0)
    # trial_summary = trial_summary[chirp_notes['good'] == 1]
    trial_mask = chirp_notes['good'] == 1

    all_marcov_matrix = []
    all_event_counts = []
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

        event_times = []
        event_labels = []
        loop_times = [chirp_times[1], rise_times[1], chirp_times[0], rise_times[0], ag_on_off_t_GRID[:, 0],
                      ag_on_off_t_GRID[:, 1], contact_t_GRID]
        loop_labels = [r'chirp$_{lose}$', r'rise$_{lose}$', r'chirp$_{win}$', r'rise$_{win}$', r'chace$_{on}$', r'chace$_{off}$', 'contact']
        event_counts = np.array([len(chirp_times[1]), len(rise_times[1]), len(chirp_times[0]), len(rise_times[0]), len(ag_on_off_t_GRID), len(ag_on_off_t_GRID), len(contact_t_GRID)])
        for ll, t in zip(loop_labels, loop_times):
            event_times.extend(t)
            event_labels.extend(np.full(len(t), ll))

        time_sorter = np.argsort(event_times)
        event_times = np.array(event_times)[time_sorter]
        event_labels = np.array(event_labels)[time_sorter]

        marcov_matrix = np.zeros((len(loop_labels), len(loop_labels)))

        for enu_ori, label_ori in enumerate(loop_labels):
            for enu_tar, label_tar in enumerate(loop_labels):
                n = len(event_times[:-1][(event_labels[:-1] == label_ori) & (event_labels[1:] == label_tar) & (np.diff(event_times) <= 5)])
                marcov_matrix[enu_ori, enu_tar] = n


        ### get those cases where ag_on does not point to event and no event points to corresponding ag_off ... add thise cases in marcov matrix
        chase_on_idx = np.where(event_labels == loop_labels[4])[0]
        chase_off_idx = np.where(event_labels == loop_labels[5])[0]
        helper_mask = np.ones_like(chase_on_idx)
        helper_mask[np.diff(event_times)[chase_on_idx] <= 5] = 0
        helper_mask[np.diff(event_times)[chase_off_idx-1] <= 5] = 0

        marcov_matrix[4, 5] += np.sum(helper_mask)

        all_marcov_matrix.append(marcov_matrix)
        all_event_counts.append(event_counts)
        # plot_transition_matrix(marcov_matrix, loop_labels)

        # plot_transition_diagram(marcov_matrix, loop_labels, node_size=event_counts)
        # plot_transition_diagram(marcov_matrix / event_counts.reshape(len(event_counts), 1) * 100, loop_labels, node_size=event_counts)
    all_marcov_matrix = np.array(all_marcov_matrix)
    all_event_counts = np.array(all_event_counts)

    collective_marcov_matrix = np.sum(all_marcov_matrix, axis=0)
    collective_event_counts = np.sum(all_event_counts, axis=0)

    plot_transition_matrix(collective_marcov_matrix, loop_labels)
    plot_transition_diagram(collective_marcov_matrix / collective_event_counts.reshape(len(collective_event_counts), 1) * 100,
                            loop_labels, collective_event_counts, threshold=5, save_str='markov_all')

    # for i in range(len(all_marcov_matrix)):
    #     plot_transition_diagram(
    #         all_marcov_matrix[i] / all_event_counts[i].reshape(len(all_event_counts[i]), 1) * 100,
    #         loop_labels, all_event_counts[i], threshold=5)
    embed()
    quit()
    pass


if __name__ == '__main__':
    main(sys.argv[1])
