import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="darkgrid")


def visualize_z_distribution(z_data, out_fn):
    sns_df = pd.DataFrame(z_data)
    sns_df.columns = ['turn_number', 'z_val']
    agg_data = []
    for turn_number in range(1, int(max(sns_df['turn_number']))+1):
        t_selected = sns_df['turn_number'] == turn_number
        agg_data.append((turn_number, np.mean(sns_df[t_selected]['z_val'])))

    agg_fd = pd.DataFrame(agg_data)
    agg_fd.columns = ['turn_number', 'z_mean_val']

    sns.boxplot(x=sns_df['turn_number'], y=sns_df['z_val'], fliersize=0, linewidth=0.5)
    sns.swarmplot(x=sns_df['turn_number'], y=sns_df['z_val'], color='.25', size=2.5)
    plt.savefig(out_fn)


def read_z_data(z_data_fd):
    turn_idx = 0
    z_data = []
    for line in z_data_fd:
        turn_idx += 1
        line = line.strip()
        if len(line) == 0:
            turn_idx = 0
            continue
        z_data.append([int(turn_idx), float(line)])
    return np.array(z_data)


def main(args):
    with open(args.z_samples_file, 'rt') as z_post_fd:
        data = read_z_data(z_post_fd)
        visualize_z_distribution(data, args.output_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_samples_file', type=str)
    parser.add_argument('--output_img', type=str)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(args)
