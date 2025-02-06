import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .utils import change_keys


def proxy_corr(metrics, top_epoch, dataset_name):
    # Compute Kendall correlations among metrics at their best epoch
    metric_list = list(metrics.keys())
    kendall_corr_values = []
    for m1 in metric_list:
        if m1 != 'final_val_acc':
            for m2 in metric_list:
                if m2 != 'final_val_acc':
                    data_m1 = metrics[m1][top_epoch[m1]]
                    data_m2 = metrics[m2][top_epoch[m2]]
                    if m1 == 'loss' and m2 == 'loss':
                        corr = stats.kendalltau(data_m1, data_m2)[0]
                    elif m1 == 'loss':
                        corr = stats.kendalltau([-v for v in data_m1], data_m2)[0]
                    elif m2 == 'loss':
                        corr = stats.kendalltau(data_m1, [-v for v in data_m2])[0]
                    else:
                        corr = stats.kendalltau(data_m1, data_m2)[0]
                    kendall_corr_values.append(corr)

    # Rename keys (e.g., from 'zico' -> 'ZiCo', etc.)
    metrics = change_keys(metrics)
    # The correlation matrix still uses the old metric_list (not yet renamed)
    # but we can proceed with a rename approach or keep consistent with top_epoch keys.
    # For simplicity, let's keep the old naming when building the matrix,
    # then rename in the final step.

    size = len(metric_list) - 1
    kendall_matrix = np.array(kendall_corr_values).reshape(size, size)

    # Sort metrics by mean correlation (excluding diagonal = 1.0)
    mean_scores = np.mean(np.ma.masked_where(kendall_matrix == 1.0, kendall_matrix), axis=1)
    sorted_indices = np.argsort(mean_scores)[::-1]
    sorted_names = [metric_list[i] for i in sorted_indices]
    sorted_matrix = kendall_matrix[sorted_indices, :][:, sorted_indices]

    names_to_remove = ['Flops']

    # Apply these replacements after sorting
    shortened_names = []
    keep_indices = []

    # Convert old metric_list to new mappings where possible
    # This block merges old->new logic so that the correlation matrix columns line up:
    mapping_after_rename = {}
    for old_key in metric_list:
        # Use change_keys for single-key dictionary
        renamed = change_keys({old_key: None})
        new_key = list(renamed.keys())[0]  # The new name or original if no rename found
        mapping_after_rename[old_key] = new_key

    # Build final name list, remove unwanted names
    for i, old_name in enumerate(sorted_names):
        new_name = mapping_after_rename[old_name]
        if new_name not in names_to_remove:
            keep_indices.append(i)
            shortened_names.append(new_name)

    filtered_matrix = sorted_matrix[keep_indices, :][:, keep_indices]

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(filtered_matrix, cmap=plt.cm.Blues)

    tick_label_font_size = 20
    text_annotation_font_size = 16

    ax.set_xticks(np.arange(len(shortened_names)))
    ax.set_yticks(np.arange(len(shortened_names)))
    ax.set_xticklabels(shortened_names, fontsize=tick_label_font_size, fontweight='bold')
    ax.set_yticklabels(shortened_names, fontsize=tick_label_font_size, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    # Gridlines
    ax.set_xticks(np.arange(len(shortened_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(shortened_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    # Cell annotations
    for i in range(len(shortened_names)):
        for j in range(len(shortened_names)):
            val = filtered_matrix[i, j]
            color = "w" if val > np.mean(filtered_matrix) else "k"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", color=color,
                fontsize=text_annotation_font_size, fontweight='bold'
            )

    plt.tight_layout()

    # (I) Save figure with dataset-based naming
    out_file = f"cross_corr_{dataset_name}.pdf"
    plt.savefig("output/" + out_file)
    print(f"Figure saved: {out_file}")
    # Uncomment if you want an interactive window:
    # plt.show()
