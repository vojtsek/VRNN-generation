import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


sns.set(style="darkgrid")


def visualize_z_distribution(z_data, out_dir):

    def _plot(x, y, mean_df, fn):
        plt.clf()
        sns.boxplot(x=x, y=y, fliersize=0, linewidth=0.5, palette='pastel', width=.3, whis=3)
        sns.swarmplot(x=x, y=y, color='.25', size=5, alpha=.2)
        sns.lineplot(x="turn_number", y="z_mean_val", data=mean_df, palette='pastel')
        plt.savefig(fn)

    sns_df = pd.DataFrame(z_data)

    agg_data = [[] for _ in sns_df.columns[1:]]
    for turn_number in range(1, int(max(sns_df[0])) + 1):
        t_selected = sns_df[0] == turn_number
        for i, a_data in enumerate(agg_data):
            a_data.append((float(turn_number)-1, np.mean(sns_df[t_selected][i+1])))

    for i, a_data in enumerate(agg_data):
        df = pd.DataFrame(a_data)
        df.columns = ['turn_number', 'z_mean_val']
        _plot(sns_df[0], sns_df[i+1], df, os.path.join(out_dir, f'z_coord{i+1}.png'))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(sns_df['turn_number'], sns_df['z_val_1'], sns_df['z_val_2'], zdir='y', c='red', s=100)
    ax.view_init(30, 90)
    plt.savefig(os.path.join(out_dir, 'z_3d.png'))
    #
    # fig = px.scatter_3d(sns_df, x='turn_number', y='z_val_1', z='z_val_2',
    #           color='turn_number')
    # fig.show()


def read_z_data(z_data_fd, dial_len=None):
    turn_idx = 0
    z_data = []
    current_dial_cache = []
    for line in z_data_fd:
        turn_idx += 1
        line = line.strip()
        if len(line) == 0:
            if len(current_dial_cache) > 0 and (dial_len is None or len(current_dial_cache) == dial_len):
                z_data.extend(current_dial_cache)
            turn_idx = 0
            current_dial_cache = []
            continue
        line = line.split()
        current_dial_cache.append([int(turn_idx)] + [float(z) for z in line])
    return np.array(z_data)


def main(args):
    z_samples_file = os.path.join(args.source_dir, 'z_posterior.txt')
    with open(z_samples_file, 'rt') as z_post_fd:
        data = read_z_data(z_post_fd, dial_len=5)
        visualize_z_distribution(data, args.source_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(args)
