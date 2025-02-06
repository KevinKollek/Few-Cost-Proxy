import os
import argparse
import json
import numpy as np
import torch
import shutil
import autoPyTorch.core.autonet_classes as autoPT


def evaluate_config(run_id, config_dir, config, logdir, sub_dir, proxy_names, device='cpu'):
    torch.backends.cudnn.benchmark = True

    logdir = os.path.join(logdir, "run_" + str(run_id) + "_" + sub_dir)

    autonet_config = {
        "min_workers": 1,
        "budget_type": "epochs",
        "default_dataset_download_dir": "./nasbench301/datasets/",
        "images_root_folders": ["./nasbench301/datasets/cifar10/"],
        "train_metric": "accuracy",
        "additional_metrics": ["cross_entropy"],
        "validation_split": 0.2,
        "use_tensorboard_logger": True,
        "networks": ['darts'],
        "images_shape": [3, 32, 32],
        "log_level": "debug",
        "random_seed": 1,
        "run_id": str(run_id),
        "result_logger_dir": logdir,
        "proxy_names": proxy_names,
        "dataloader_worker": 2,
        "device": device}

    # Initialize
    autonet = autoPT.AutoNetImageClassificationMultipleDatasets(**autonet_config)

    if isinstance(config_dir, dict):
        hyperparameter_config = config_dir['optimized_hyperparamater_config']

    else:
        if config is None:
            # Read hyperparameter config
            with open(config_dir, "r") as f:
                hyperparameter_config = json.load(f)
        else:
            with open(config, "r") as f:
                hyperparameter_config = json.load(f)

    hyperparameter_config["NetworkSelectorDatasetInfo:network"] = "darts"
    print(hyperparameter_config)
    budget = hyperparameter_config['SimpleLearningrateSchedulerSelector:cosine_annealing:T_max']

    autonet_config = autonet.get_current_autonet_config()
    result = autonet.refit(X_train=np.array([os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "datasets/CIFAR10.csv")]), Y_train=np.array([0]),
                           X_valid=None, Y_valid=None,
                           hyperparameter_config=hyperparameter_config,
                           autonet_config=autonet_config,
                           budget=budget, budget_type="epochs")

    print("Done with refitting.")

    torch.cuda.empty_cache()

    with open(str(logdir + "/proxies_dict.json"), 'w+') as f:
        json.dump(result['proxies'], f)

    # Dump
    with open(os.path.join(logdir, "final_output.json"), "w+") as f:
        json.dump(result, f)

    return result


def get_config_dir(config_parent_dir, run_id):
    config_dirs = [os.path.join(config_parent_dir, p) for p in os.listdir(config_parent_dir) if p.startswith("config_")]
    config_dirs.sort(key=lambda x: int(x.replace(".json", "").split("_")[-1]))
    return config_dirs[run_id - 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit a random config on CIFAR task')
    parser.add_argument("--run_id", type=int, help="An id for the run.")
    parser.add_argument("--config_parent_dir", type=str, help="Path to config.json", default="./configs")
    parser.add_argument("--config", type=str, help="Config as json string", default=None)
    parser.add_argument("--logdir", type=str, help="Directory the results are written to.",
                        default="nasbench301/logs/darts_proxy")
    parser.add_argument("--offset", type=int, help="An id for the run.", default=0)
    parser.add_argument("--budget", type=int, help="number of epochs", default=0)
    parser.add_argument("--load", type=int, default=(0, 3), nargs='+', help="An id for the run.")
    parser.add_argument("--sub_dir", type=str, default="rs", help="An id for the run.")
    parser.add_argument("--save_dir", type=str, help="Directory the results are written to.", default="output/")
    parser.add_argument('--proxy_names', type=str,
                        default=['val_acc', 'zico', 'grad_norm', 'absmaginc', 'movement', 'large_final',
                                 'magnitude_increase', 'synflow', 'snip', 'grasp', 'l2_norm', 'jacov', 'plain',
                                 'fisher', 'nwot', 'zen', 'epe_nas'], nargs='+', help='The applied datasets.')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()
    run_id = args.run_id + args.offset
    sub_dir = args.sub_dir
    all_proxies = {}

    if os.path.exists(str(args.save_dir + f"/proxy_dict_nb301_cf10_{args.load[1]}_archs.json")):
        with open(str(args.save_dir + f"/proxy_dict_nb301_cf10_{args.load[1]}_archs.json"), 'r') as file:
            all_proxies = json.load(file)

    data_dir = r'./nasbench301/datasets/nb_301_v13_lc_iclr_final/'

    rundir = "run_" + str(run_id) + "_" + str(sub_dir)
    try:
        with open(str(data_dir + sub_dir + "/results_" + str(run_id) + ".json"), "r+") as read_file:
            config_dir = json.load(read_file)
    except:
        with open(str(data_dir + sub_dir + "/results_" + str(run_id) + "_ti.json"), "r+") as read_file:
            config_dir = json.load(read_file)

    config_dir['optimized_hyperparamater_config'][
        'SimpleLearningrateSchedulerSelector:cosine_annealing:T_max'] = args.budget

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if rundir in os.listdir(args.logdir):
        subdirs = os.listdir(os.path.join(args.logdir, rundir))
        if ("final_output.json" in subdirs) and ("bohb_status.json" in subdirs):
            pass
        else:
            shutil.rmtree(os.path.join(args.logdir, rundir))
            result = evaluate_config(
                run_id=run_id,
                config_dir=config_dir,
                config=args.config,
                logdir=args.logdir,
                sub_dir=sub_dir,
                proxy_names=args.proxy_names,
                device=device
            )
            all_proxies[str(run_id)] = result['proxies']
    else:
        result = evaluate_config(
            run_id=run_id,
            config_dir=config_dir,
            config=args.config,
            logdir=args.logdir,
            sub_dir=sub_dir,
            proxy_names=args.proxy_names,
            device=device
        )
        all_proxies[str(run_id)] = result['proxies']

    with open(str(args.save_dir + f"/proxy_dict_nb301_cf10_{args.load[1]}_archs.json"), 'w+') as f:
        json.dump(all_proxies, f)
