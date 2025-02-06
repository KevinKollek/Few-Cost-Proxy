import copy
import json
import math
import os
from math import log2

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random
import matplotlib.ticker as ticker


def fill_lists_in_dict(data):
    """
    Ensures that all lists (at every level within the dictionary) are extended
    to the maximum list length found in the same dictionary scope.
    Uses recursion to handle nested dictionaries.

    :param data: Dictionary (potentially nested) with list values.
    :return: The same dictionary with all lists padded to equal lengths.
    """
    # Collect direct lists and sub-dicts in the current dictionary level
    lists_in_this_dict = []
    sub_dicts = []

    for key, value in data.items():
        if isinstance(value, dict):
            sub_dicts.append(value)
        elif isinstance(value, list):
            lists_in_this_dict.append(value)

    # Pad all lists in this dictionary level to the max length
    if lists_in_this_dict:
        max_length = max(len(lst) for lst in lists_in_this_dict)
        for lst in lists_in_this_dict:
            if len(lst) < max_length:
                # Extend current list to match the max_length
                lst.extend([lst[-1]] * (max_length - len(lst)))

    # Recursively fill lists for sub-dicts
    for sub_d in sub_dicts:
        fill_lists_in_dict(sub_d)

    return data


def final_val_acc_correlation(
        data,
        epochs,
        threshold=0.4,
        random_seeds=1,
        total_samples=1000,
        plot_highest_matrix=True,
        plot_graph=True,
        plot_legend=True,
        row=5,
        column=5,
        corr_type='spearman'
):
    """
        Calculates correlations of various metrics (e.g., loss) with 'final_val_acc'
        across different epochs. It then plots several outputs (e.g., heatmaps of
        the highest correlations, correlation vs. epochs).

        :param data: Dictionary containing architectures and metrics
        :param epochs: Number of epochs to consider
        :param threshold: Minimum correlation threshold for filtering metrics in plots
        :param random_seeds: Number of random seeds to create different subsets of data
        :param total_samples: The size of each randomly generated data subset
        :param plot_highest_matrix: If True, plots a heatmap of the highest correlations
        :param plot_graph: If True, plots the correlation values over epochs
        :param plot_legend: If True, includes a legend in the correlation plot
        :param row: Number of rows in the heatmap layout
        :param column: Number of columns in the heatmap layout
        :param corr_type: Type of correlation to use: 'spearman' or 'kendall'
        """

    not_allowed_metrics = ['final_val_acc', 'final_test_acc']

    data = fill_lists_in_dict(data)

    if len(list(list(data.values())[0].values())[0]) < epochs:
        epochs = len(list(list(data.values())[0].values())[0])

    if total_samples > len(data):
        total_samples = len(data)

    # List to store the generated dictionaries
    generated_dicts = {}

    # Generate 100 different dictionaries
    for seed_value in range(random_seeds):
        # Set the random seed
        random.seed(seed_value)

        # Create a new dictionary by copying the original dictionary
        new_dict = dict(data)

        # Randomly select and remove five elements from the new dictionary
        selected_items = random.sample(list(data.items()), len(data) - total_samples)
        for key, _ in selected_items:
            del new_dict[key]

        # Add the generated dictionary to the list
        generated_dicts[seed_value] = new_dict

    for i in range(len(generated_dicts)):
        data = generated_dicts[i]

        correlation = {}

        highest_correlation = {}
        highest_kendall = {}

        position_corr = {}
        # Iterieren durch jedes Element und Erstellen des entsprechenden Plots
        for metric in data[list(data.keys())[0]]:

            if metric not in not_allowed_metrics:
                highest_correlation[metric] = (-1, -1)
                position_corr[metric] = []
                correlation[metric] = []
                for pos in range(epochs):
                    metrics = []
                    accs = []
                    for arch in data:
                        if isinstance(data[arch][metric], list):
                            metrics.append(data[arch][metric][pos])

                        else:
                            metrics.append(float(data[arch][metric]))
                        accs.append(data[arch]['final_val_acc'])

                    for idx_metric, sample_metric in enumerate(metrics):
                        if math.isnan(sample_metric):
                            metrics[idx_metric] = -100000000

                    if corr_type == "spearman":
                        correlation[metric].append(stats.spearmanr(accs, metrics)[0])
                    elif corr_type == "kendall":
                        correlation[metric].append(stats.kendalltau(accs, metrics)[0])

                    if "loss" in list(correlation.keys()):
                        correlation['loss'] = [-i for i in correlation['loss']]

                    if highest_correlation[metric][1] < correlation[metric][-1]:
                        highest_correlation[metric] = (pos, correlation[metric][-1])

                    position_corr[metric].append(
                        0 if correlation[metric][-1] is np.NAN else round(correlation[metric][-1], 3))

        filtered_spearman = copy.deepcopy(position_corr)
        for key, value in highest_correlation.items():
            highest_correlation[key] = (highest_correlation[key][0], round(highest_correlation[key][1], 3))
            if highest_correlation[key][1] < threshold:
                del filtered_spearman[key]

        # Create a heatmap
        if plot_highest_matrix:
            fig, ax = plt.subplots(figsize=(15, 15))
            im = ax.imshow([[highest_correlation[list(highest_correlation.keys())[i * row + j]][1] if i * row + j < len(
                highest_correlation) else 0 for j in range(row)] for i in range(column)], cmap=plt.cm.Blues)
            cbar = ax.figure.colorbar(im, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Spearman Correlation, Samples {:}".format(total_samples))
            # Loop over the data and add annotations to the heatmap
            for i in range(column):
                for j in range(row):
                    if i * row + j < len(position_corr):
                        ax.text(j, i, "{}\n{}\n{}".format(
                            highest_correlation[list(highest_correlation.keys())[i * row + j]][1],
                            highest_correlation[list(highest_correlation.keys())[i * row + j]][0],
                            list(highest_correlation.keys())[i * row + j]), ha="center",
                                va="center", color="black", fontsize=16)

        if plot_graph:
            fig_grph, ax_grph = plt.subplots()
            for key, value in filtered_spearman.items():
                ax_grph.plot(value, label=key)
            ax_grph.set_xlabel("Epochs")
            ax_grph.set_ylabel("Correlation")
            ax_grph.set_title("Spearman Correlation")

            if "movement" in position_corr:
                ax_grph.set_xticks([i for i in range(len(position_corr["movement"]))])
            if plot_legend:
                ax_grph.legend()

        if plot_graph or plot_highest_matrix:
            # Show the plot
            plt.tight_layout()
            plt.show()

    top_epoch = dict()
    for proxy in highest_correlation:
        top_epoch[proxy] = highest_correlation[proxy][0]

    return top_epoch
