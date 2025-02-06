import argparse, os
from nasbench201.procedures.train import train_models

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    parser = argparse.ArgumentParser(description='NAS-Bench-201',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Setup
    parser.add_argument('--save_dir', type=str, default='output/',
                        help='Folder to save checkpoints and log.')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 2)')

    # NAS-Bench-201 Settings
    parser.add_argument('--max_node', type=int, default=4, help='The maximum node in a cell.')
    parser.add_argument('--num_cells', type=int, default=5, help='The number of cells in one stage.')
    parser.add_argument('--channel', type=int, default=16, help='The number of channels.')

    # Dataset Settings
    parser.add_argument('--archnumber', type=int, default=1000, help='The number of models to be trained and evaluated')
    parser.add_argument('--seed', type=int, default=777, nargs='+', help='The range of models to be evaluated')

    parser.add_argument('--dataset', type=str, default='cifar10-valid',
                        help='The applied datasets.')  # cifar10-valid, cifar100, syn_cifar10, scifar100, nb301_cifar10

    # Evaluation Settings
    parser.add_argument('--proxy_names', type=str,
                        default=['val_acc', 'zico', 'grad_norm', 'absmaginc', 'movement', 'large_final',
                                 'magnitude_increase', 'synflow', 'snip', 'grasp', 'l2_norm', 'jacov', 'plain',
                                 'fisher', 'nwot', 'zen', 'epe_nas'], nargs='+', help='The applied datasets.')

    # Training Settings
    parser.add_argument('--use_bn', default=True, help='Enable/Disable BatchNorm', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=9, help='Number of epochs to be trained')

    args = parser.parse_args()

    if args.dataset == 'nb301_cifar10':
        import subprocess, sys
        if sys.platform == "win32":
            python_executable = os.path.join(os.environ["CONDA_PREFIX"], "python.exe")
        else:
            python_executable = os.path.join(os.environ["CONDA_PREFIX"], "bin", "python")

        for i in range(args.archnumber):
            command = [
                python_executable, "nasbench301/run_proxy.py",
                "--run_id", str(i),
                "--load", str(0), str(args.archnumber),
                "--sub_dir", "rs",
                "--budget", str(args.max_epochs),
                "--save_dir", str(args.save_dir),
                "--proxy_names", *args.proxy_names
            ]

            subprocess.run(command)
    else:
        train_models(args)


if __name__ == '__main__':
    main()
