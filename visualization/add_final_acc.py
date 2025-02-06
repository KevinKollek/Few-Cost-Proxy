import json
import os

def _update_metrics(measures_dict, arch_key, result, dataset):
    """Helper function to update final metrics in measures_dict for a single architecture."""
    measures_dict[arch_key]['final_val_acc'] = result.eval_acc1es['ori-test@199']

    # For CIFAR-100, store additional validation/test metrics
    if dataset == 'cifar100':
        measures_dict[arch_key]['final_x-valid'] = result.eval_acc1es['x-valid@199']
        measures_dict[arch_key]['final_x-test'] = result.eval_acc1es['x-test@199']


def extract_final_val_acc_nasbench201(api, measures_dict, dataset):
    for arch_key, arch_metrics in measures_dict.items():
        # Only update if certain keys are missing or some other logic is needed
        if (
                'final_val_acc' not in arch_metrics
                or 'params' not in arch_metrics
                or 'flop' not in arch_metrics
        ):
            # Loop through each architecture in the API to find a match
            for _, arch_info in api.arch2infos_dict.items():
                arch_data = arch_info['200']
                if arch_key in arch_data.arch_str:
                    all_results = arch_data.all_results
                    if (dataset, 777) in all_results:
                        _update_metrics(measures_dict, arch_key, all_results[(dataset, 777)], dataset)
                    elif (dataset, 888) in all_results:
                        _update_metrics(measures_dict, arch_key, all_results[(dataset, 888)], dataset)
                    else:
                        _update_metrics(measures_dict, arch_key, all_results[(dataset, 999)], dataset)
                    break
    return measures_dict


def extract_final_val_acc_nasbench301(data):
    result_config = {}
    sub_dir = "./nasbench301/datasets/nb_301_v13_lc_iclr_final/rs"
    for run_id in range(1000):
        try:
            with open(str(sub_dir + "/results_" + str(run_id) + ".json"), "r+") as read_file:
                result_config[str(run_id)] = json.load(read_file)
        except:
            with open(str(sub_dir + "/results_" + str(run_id) + "_ti.json"), "r+") as read_file:
                result_config[str(run_id)] = json.load(read_file)



    for key in data:
        data[key]['final_val_acc'] = result_config[key]['test_accuracy']

    return data

