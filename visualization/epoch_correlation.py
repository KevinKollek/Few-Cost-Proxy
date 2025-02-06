import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def epoch_corr(result_dict, dataset_name):
    # Compute Kendall correlations between consecutive epochs for each proxy.
    kendall_corr = {}
    for proxy_name, epoch_values in result_dict.items():
        if "final_val_acc" == proxy_name:
            continue
        kendall_corr[proxy_name] = []
        # Calculate correlation between consecutive epochs
        for i in range(len(epoch_values) - 1):
            tau = stats.kendalltau(epoch_values[i], epoch_values[i + 1])[0]
            kendall_corr[proxy_name].append(tau)

    # Convert correlation data into a table suitable for plotting.
    # Only the first 9 consecutive pairs are used (idx < 9).
    data_rows = []
    proxy_names = list(kendall_corr.keys())
    for proxy in proxy_names:
        row = [round(val, 3) for idx, val in enumerate(kendall_corr[proxy]) if idx < 9]
        data_rows.append(row)

    data_for_imshow = np.array(data_rows)

    # Sort the columns by mean to get a more organized heatmap.
    def sort_by_mean(df):
        return df.reindex(df.mean().sort_values().index, axis=1)

    # Build a DataFrame for the heatmap.
    predictors_map = {i: list(kendall_corr.keys())[i] for i in range(len(list(kendall_corr.keys())))}
    index_dict = {
        0: '(0,1)', 1: '(1,2)', 2: '(2,3)', 3: '(3,4)',
        4: '(4,5)', 5: '(5,6)', 6: '(6,7)', 7: '(7,8)', 8: '(8,9)'
    }

    gain_df = pd.DataFrame(data_for_imshow)
    gain_df = gain_df.rename(columns=index_dict, index=predictors_map).transpose()
    gain_df = sort_by_mean(gain_df)

    # Heatmap plotting function.
    def plot_heatmap(df, figsize=(10, 5), rotation=40, title='',
                     cmap='viridis_r', savetitle='zcp_corr', square=False, fmt='.3f'):
        plt.figure(figsize=figsize, dpi=200)
        # If you want a title on the heatmap, uncomment the next line.
        # plt.title(title, fontsize=15)

        ax = sns.heatmap(
            df, annot=True, cmap=cmap, fmt=fmt, cbar=False,
            square=square, annot_kws={"size": 10}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, fontsize=12, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12, va='center')
        ax.set(xlabel=None, ylabel=None)

        plt.tight_layout()
        plt.savefig(f'output/{savetitle}_{dataset_name}.pdf', bbox_inches='tight')
        plt.show()  # Optional: show the plot interactively

    # Generate and save the heatmap.
    plot_heatmap(
        gain_df,
        title='Few-Cost Proxy correlation between epochs on TSS-Cf10',
        figsize=(10, 5),
        savetitle='epoch_correlation'
    )
