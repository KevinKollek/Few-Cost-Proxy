from pathlib import Path
import os
import argparse
import json
from visualization.epoch_correlation import epoch_corr
from visualization.proxy_correlation import proxy_corr
from visualization.corr_vis import final_val_acc_correlation
from visualization.add_final_acc import extract_final_val_acc_nasbench201, extract_final_val_acc_nasbench301

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# File paths for each dataset
DATA_PATHS = {
    "nb201_c10": [
        "proxy_dict_cifar10-valid_1000_archs.json"
    ],
    "nb201_c100": [
        "proxy_dict_cifar100_1000_archs.json"
    ],
    "nb201_sync10": [
        "proxy_dict_syn_cifar10_1000_archs.json"
    ],
    "nb201_sphc100": [
        "proxy_dict_scifar100_1000_archs.json"
    ],
    "nb301_c10": [
        "proxy_dict_nb301_cf10_1000_archs.json"
    ],
}


def visualize_dataset(args):
    """
    Loads the dataset, computes Kendall correlations at chosen epochs, sorts them,
    and plots a heatmap. Saves the figure to a file based on the dataset name.
    """

    dataset_name = args.dataset
    path = args.path
    epochs = args.epochs
    corr_thresh = args.corr_thresh

    # Load the JSON file(s)
    data_files = DATA_PATHS[dataset_name]
    full_paths = [f"{path}/{data_file}" for data_file in data_files]
    with open(full_paths[0], "r") as f:
        data = json.load(f)

    epochs = min(epochs, len(data[list(data.keys())[0]][list(data[list(data.keys())[0]].keys())[0]]))

    if "nb301" in dataset_name:
        extract_final_val_acc_nasbench301(data)
    elif dataset_name == "nb201_c10" or dataset_name == "nb201_c100":
        from nasbench201.nas_201_api import NASBench201API
        meta_file = Path("nasbench201/nas_201_api/NAS-Bench-201-v1_1-096897.pth")
        api = NASBench201API(meta_file, verbose=True)
        if dataset_name == "nb201_c10":
            data = extract_final_val_acc_nasbench201(api, data, 'cifar10-valid')
        elif dataset_name == "nb201_c100":
            data = extract_final_val_acc_nasbench201(api, data, 'cifar100')
    elif dataset_name == "nb201_sync10":
        with open("trained_networks/synthetic_cifar10_final_val_acc.json", "r") as file:
            final_acc = json.load(file)

        for key, value in data.items():
            if key in final_acc.keys():
                data[key]['final_val_acc'] = final_acc[key]['final_val_acc']
    elif dataset_name == "nb201_sphc100":
        with open("trained_networks/spherical_cifar100_final_val_acc.json", "r") as file:
            final_acc = json.load(file)

        for key, value in data.items():
            if key in final_acc.keys():
                data[key]['final_val_acc'] = final_acc[key]['final_val_acc']

    for key, value in data.items():
        if dataset_name == "nb201_c100" and "final_x-valid" in value:
            del data[key]["final_x-valid"]

        if dataset_name == "nb201_c100" and "final_x-test" in value:
            del data[key]["final_x-test"]

    top_epoch = final_val_acc_correlation(data, epochs, corr_thresh)

    # Collect metrics into a dict of shape: metrics[metric_name][epoch_index] = list of values
    metrics = {}

    # For the first architecture in data, check what metrics exist
    first_arch = list(data.keys())[0]
    for metric in data[first_arch]:
        metrics[metric] = [[] for _ in range(epochs)]
        for arch in data:
            # Handle scalar vs. list
            if isinstance(data[arch][metric], (float, str)):
                val_as_float = float(data[arch][metric])
                for pos in range(epochs):
                    metrics[metric][pos].append(val_as_float)
            else:
                for pos in range(epochs):
                    metrics[metric][pos].append(data[arch][metric][pos])

    # # Rename "param" to "params" if present
    # if "param" in metrics:
    #     metrics["params"] = metrics["param"]
    #     del metrics["param"]

    # If the dataset is nb301_c10, you may want to remove "final_test_acc"
    # (Depending on your previous logic)
    if dataset_name == "nb301_c10" and "final_test_acc" in metrics:
        del metrics["final_test_acc"]

    proxy_corr(metrics, top_epoch, dataset_name)
    epoch_corr(metrics, dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Visualize NASBench Datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="nb201_c10",
        choices=["nb201_c10", "nb201_c100", "nb201_sync10", "nb201_sphc100", "nb301_c10"],
        help="Select which dataset/search space to visualize."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="output",
        help="Specify the path to the dataset files."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Specify the number of epochs to run."
    )
    parser.add_argument(
        "--corr_thresh",
        type=float,
        default=0.5,
        help="Specify the max threshold for visualizing correlations."
    )
    args = parser.parse_args()

    visualize_dataset(args)


if __name__ == "__main__":
    main()
