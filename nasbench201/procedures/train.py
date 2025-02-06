import torch
import json
import copy
from nasbench201.models import CellStructure, get_cell_based_tiny_net
from nasbench201.dataset import select_dataset
from nasbench201.procedures.utils import set_random_seed
from nasbench201.procedures   import prepare_seed, get_optim_scheduler
from nasbench201.configs.config_utils import dict2config
from nasbench201.utils        import get_model_infos, AverageMeter, obtain_accuracy
from nasbench201.procedures.functions import calculate_accuracy, find_proxies
from tqdm import tqdm

def train_models(args):
    # Initialization
    torch.set_num_threads(args.workers)
    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load all architectures
    if args.dataset == 'cifar10-valid' or args.dataset == 'cifar100':
        all_archs = json.load(open("nasbench201/dataset/all_archs.json", "r", encoding="utf-8"))
    elif args.dataset == 'syn_cifar10':
        all_archs = json.load(open("nasbench201/dataset/syn_archs.json", "r", encoding="utf-8"))
        args.workers = 0
    elif args.dataset == 'scifar100':
        all_archs = json.load(open("nasbench201/dataset/sph_archs.json", "r", encoding="utf-8"))

    arch_config = {
        'channel': args.channel,
        'num_cells': args.num_cells,
        'use_bn': args.use_bn
    }

    proxy_dict = {}
    to_evaluate_indexes = list(range(args.archnumber))

    # Load dataset
    train_loader, val_loaders, config = select_dataset(args.dataset, args.workers)

    # Iterate over architectures
    for idx in to_evaluate_indexes:
        prepare_seed(args.seed)
        arch = all_archs[idx]

        print("Architecture "+str(idx+1)+"/"+str(args.archnumber))
        print(arch)

        # Build network
        network_dict = {
            'name': 'infer.tiny',
            'C': arch_config['channel'],
            'N': arch_config['num_cells'],
            'genotype': CellStructure.str2structure(arch),
            'num_classes': config.class_num,
            'use_bn': arch_config['use_bn']
        }
        net = get_cell_based_tiny_net(dict2config(network_dict))
        net.to(device)
        net.network_dict = network_dict

        # Get flop and param info (stored for later use)
        flop, param = get_model_infos(net, config.xshape)

        # Set up optimizer, scheduler, and loss
        optimizer, scheduler, criterion = get_optim_scheduler(net.parameters(), config)

        # Keep an initial copy of network parameters if needed for proxy calculations
        net.param_list = []
        for _, p in net.named_parameters():
            net.param_list.append(copy.deepcopy(p))

        # Dictionary to store proxy values across epochs
        all_epoch_proxies = {}

        # Training loop over epochs
        for epoch in range(args.max_epochs):
            scheduler.update(epoch, 0.0)
            net.train()

            losses_meter = AverageMeter()
            top1_meter = AverageMeter()

            inputs_list, targets_list = [], []
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.max_epochs}')

            for i, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                inputs_list.append(inputs)
                targets_list.append(targets)

                if epoch == 0 and i<2:
                    continue

                elif epoch == 0 and i == 2:
                    epoch_proxies = find_proxies(
                        net,
                        inputs_list,
                        targets_list,
                        proxy_names=[name for name in args.proxy_names if name != 'val_acc'],
                        criterion=criterion
                    )
                    # Compute validation accuracy if 'val_acc' is requested
                    if 'val_acc' in args.proxy_names:
                        if 'x-valid' in val_loaders:
                            val_acc = calculate_accuracy(net, val_loaders['x-valid'], device)
                        else:
                            val_acc = calculate_accuracy(net, val_loaders['ori-test'], device)
                        epoch_proxies['val_acc'] = val_acc
                    for j in range(i):
                        # Update scheduler, reset optimizer
                        scheduler.update(None, float(j) / len(train_loader))
                        optimizer.zero_grad()

                        # Forward and backward
                        _, logits = net(inputs_list[j])
                        loss = criterion(logits, targets_list[j])
                        loss.backward()
                        optimizer.step()

                        # Update metrics
                        losses_meter.update(loss.item(), inputs_list[j].size(0))
                        prec1, _ = obtain_accuracy(logits.data, targets_list[j].data, topk=(1, 5))
                        top1_meter.update(prec1.item(), inputs_list[j].size(0))

                        # Update the progress bar with current metrics
                        progress_bar.set_postfix(loss=losses_meter.avg, acc=top1_meter.avg)

                    # Save average training loss and accuracy for this epoch
                    epoch_proxies['loss'] = float(losses_meter.avg)
                    epoch_proxies['train_acc'] = top1_meter.avg
                    # Append epoch proxies to the per-architecture record
                    for key, value in epoch_proxies.items():
                        all_epoch_proxies.setdefault(key, []).append(value)
                else:

                    # Update scheduler, reset optimizer
                    scheduler.update(None, float(i) / len(train_loader))
                    optimizer.zero_grad()

                    # Forward and backward
                    _, logits = net(inputs)
                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()

                    # Update metrics
                    losses_meter.update(loss.item(), inputs.size(0))
                    prec1, _ = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
                    top1_meter.update(prec1.item(), inputs.size(0))

                    # Update the progress bar with current metrics
                    progress_bar.set_postfix(loss=losses_meter.avg, acc=top1_meter.avg)

            # After one epoch, compute proxies using the last few batches (or all if fewer than 3)
            if len(inputs_list) >= 3:
                inputs_for_proxy = inputs_list[-3:]
                targets_for_proxy = targets_list[-3:]
            else:
                inputs_for_proxy = inputs_list
                targets_for_proxy = targets_list

            epoch_proxies = find_proxies(
                net,
                inputs_for_proxy,
                targets_for_proxy,
                proxy_names=[name for name in args.proxy_names if name != 'val_acc'],
                criterion=criterion
            )

            # Compute validation accuracy if 'val_acc' is requested
            if 'val_acc' in args.proxy_names:
                if 'x-valid' in val_loaders:
                    val_acc = calculate_accuracy(net, val_loaders['x-valid'], device)
                else:
                    val_acc = calculate_accuracy(net, val_loaders['ori-test'], device)
                epoch_proxies['val_acc'] = val_acc

            # Save average training loss and accuracy for this epoch
            epoch_proxies['loss'] = float(losses_meter.avg)
            epoch_proxies['train_acc'] = top1_meter.avg

            # Append epoch proxies to the per-architecture record
            for key, value in epoch_proxies.items():
                all_epoch_proxies.setdefault(key, []).append(value)

        # Store the complete results for this architecture
        proxy_dict[arch] = {
            **all_epoch_proxies,
            'flop': flop,
            'param': param
        }

        # Save proxy dictionary to a JSON file
        out_filename = f"proxy_dict_{args.dataset}_{args.archnumber}_archs.json"
        with open(f"output/{out_filename}", 'w', encoding='utf-8') as f:
            json.dump(proxy_dict, f)