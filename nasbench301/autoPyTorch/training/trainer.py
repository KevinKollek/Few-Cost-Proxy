import time
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoints.save_load import save_checkpoint
from tqdm import tqdm

import autoPyTorch.proxies as proxies


def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def find_proxies_arrays(net_orig, inputs, targets, proxy_names=None, loss_fn=F.cross_entropy):
    if proxy_names is None:
        proxy_names = proxies.available_proxies
    torch.cuda.empty_cache()

    done, ds = False, 1
    proxy_values = {}

    while not done:
        try:
            for proxies_name in proxy_names:
                net_copy = copy.deepcopy(net_orig)
                net_copy = net_copy.cpu()
                if proxies_name not in proxy_values:
                    val = proxies.calc_proxies(proxies_name, net_copy, inputs, targets, loss_fn=loss_fn, split_data=ds)
                    proxy_values[proxies_name] = val

            done = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                done = False
                if ds == inputs[0].shape[0] // 2:
                    raise ValueError(f'Can\'t split data anymore, but still unable to run. Something is wrong')
                ds += 1
                while inputs[0].shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f'Caught CUDA OOM, retrying with data split into {ds} parts')
            else:
                raise e

    return proxy_values


def find_proxies(net_orig,  # neural network
                 inputs, targets,  # a data loader (typically for training data)
                 # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
                 criterion,  # loss function to use within the zero-cost metrics
                 proxy_names=None,
                 # an array of proxies names to compute, if left blank, all proxies are computed by default
                 proxies_arr=None):  # [not used] if the proxies are already computed but need to be summarized, pass them here

    def sum_arr(arr):
        for i in range(len(arr)):
            if len(arr[i].shape) > 1:
                arr[i] = round(float(torch.sum(arr[i])), 5)
            else:
                arr[i] = round(float(arr[i]), 5)
        return arr

    if proxies_arr is None:
        proxies_arr = find_proxies_arrays(net_orig, inputs, targets, proxy_names, loss_fn=criterion)

    return proxies_arr


def calculate_accuracy(network, data_loader, proxy_names, criterion, device):
    """
    Calculate the accuracy based on the given proxy names using the provided network, data loader, and criterion.

    Args:
        network: The network model.
        data_loader: DataLoader for the dataset.
        proxy_names: List of proxy names to calculate.
        criterion: The loss criterion.

    Returns:
        float: Average accuracy based on the given proxy names.
    """
    network.eval()
    total_accuracy = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            if device == 'cuda':
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
            else:
                inputs = inputs.cpu()
                targets = targets.cpu()

            # Forward pass
            outputs = network(inputs)

            # Calculate accuracy
            prec1, prec5 = obtain_accuracy(outputs[1].data, targets.data, topk=(1, 5))

            # Update counts
            total_accuracy += float(prec1[0])

    network.train()

    # Calculate average accuracy
    average_accuracy = total_accuracy / len(data_loader)

    return average_accuracy


class Trainer(object):
    def __init__(self, loss_computation, model, criterion, budget, optimizer, scheduler, budget_type, device,
                 images_to_plot=0, checkpoint_path=None, config_id=None):
        self.checkpoint_path = checkpoint_path
        self.config_id = config_id

        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device

        self.budget = budget
        self.loss_computation = loss_computation

        self.images_plot_count = images_to_plot

        self.budget_type = budget_type
        self.cumulative_time = 0

        self.train_loss_sum = 0
        self.train_iterations = 0

        self.latest_checkpoint = None

        print("TRAINER: Visible devices:", self.device)

        if torch.cuda.device_count() > 1:
            print("TRAINER: Found " + str(torch.cuda.device_count()) + " devices")
            model = nn.DataParallel(model)
        self.model = model.to(self.device)

        unique_devices = []
        for p in self.model.parameters():
            if p.device not in unique_devices:
                unique_devices.append(p.device)

        print("TRAINER: Model on devices:", unique_devices)

        try:
            self.criterion = criterion.to(self.device)
        except:
            print("No criterion specified.")
            self.criterion = None

    def update_proxies(self, network, inputs_list, train_loader, targets_list, recursive_loader, proxy_names, criterion,
                       old_proxies, eval_accs=False):
        """
            Update the proxies in the old_proxies dictionary based on the network's performance on recent inputs and targets.

            Args:
                network: The network model.
                inputs_list (list): A list of input tensors.
                targets_list (list): A list of target tensors.
                proxy_names (list): The list of proxy names.
                criterion: The loss criterion.
                old_proxies (dict): The dictionary containing previous proxy values.

            Returns:
                dict: The updated old_proxies dictionary with new proxy values.
            """
        if proxy_names is not None and 'val_acc' in proxy_names:
            proxy_names = [name for name in proxy_names if name != 'val_acc']

        proxies = find_proxies(network, inputs_list[-3:], targets_list[-3:], proxy_names=proxy_names,
                               criterion=criterion)

        if old_proxies is None:
            # create new dictionary with same keys, but values in lists
            old_proxies = {key: [value] for key, value in proxies.items()}
        else:
            for key, value in proxies.items():
                if key in old_proxies.keys():
                    old_proxies[key].append(value)
                else:
                    old_proxies[key] = [value]

        return old_proxies

    def train(self, epoch, train_loader, metrics, old_proxies, proxy_names):
        '''
            Trains the model for a single epoch
        '''
        loss_sum = 0.0
        N = 0

        classified = []
        misclassified = []

        print("TRAINER: training epoch", str(epoch), " budget type:", self.budget_type)

        self.model.train()

        inputs_list = []
        targets_list = []
        budget_exceeded = False
        metric_results = [0] * len(metrics)

        for step, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Progress"):
            data = data.to(self.device)
            targets = targets.to(self.device)
            inputs_list.append(data)
            targets_list.append(targets)
            if step < 2:
                continue
            elif step == 2:
                self.model.loss_computation = self.loss_computation
                self.model.criterion = self.criterion

                old_proxies = self.update_proxies(self.model, inputs_list, train_loader, targets_list, None,
                                                  proxy_names, self.criterion.function,
                                                  old_proxies)

            data, criterion_kwargs = self.loss_computation.prepare_data(data, targets)
            batch_size = data.size(0)

            aux_head = False
            outputs = self.model(data)

            # TODO: enable multiple auxiliary outputs
            if isinstance(outputs, tuple) and len(outputs) == 2:
                outputs, aux_outputs = outputs
                aux_head = True

            loss_func = self.loss_computation.criterion(**criterion_kwargs)
            loss = loss_func(self.criterion, outputs)

            if aux_head:
                loss += 0.4 * loss_func(self.criterion, aux_outputs)

            if step == 2:
                if 'loss' not in old_proxies.keys():
                    old_proxies['loss'] = [round(float(loss), 5)]

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            if self.images_plot_count > 0:
                with torch.no_grad():
                    _, pred = outputs.topk(1, 1, True, True)
                    pred = pred.t()
                    correct = pred.eq(targets.view(1, -1).expand_as(pred)).cpu().numpy()[0]
                    data = data.cpu().numpy()
                    classified += list(data[correct.astype(bool)])
                    misclassified += list(data[(1 - correct).astype(bool)])
                    if len(classified) > self.images_plot_count:
                        classified = random.sample(classified, self.images_plot_count)
                    if len(misclassified) > self.images_plot_count:
                        misclassified = random.sample(misclassified, self.images_plot_count)

            with torch.no_grad():
                for i, metric in enumerate(metrics):
                    metric_results[i] += self.loss_computation.evaluate(metric, outputs,
                                                                        **criterion_kwargs) * batch_size

            if 'train_acc' not in old_proxies.keys():
                old_proxies['train_acc'] = [round(float(metric_results[0]) / batch_size, 5)]
            if 'train_cross_entropy' not in old_proxies.keys():
                old_proxies['train_cross_entropy'] = [round(float(metric_results[1]) / batch_size, 5)]

            loss_sum += loss.item() * batch_size
            N += batch_size

            if self.budget_type == 'time' and self.cumulative_time + (time.time() - start_time) >= self.budget:
                budget_exceeded = True
                break

        if epoch == self.budget-1:
            old_proxies = self.update_proxies(self.model, inputs_list[-3:], train_loader, targets_list[-3:], None,
                                              proxy_names, self.criterion.function,
                                              old_proxies)

        if self.images_plot_count > 0:
            import tensorboard_logger as tl
            tl.log_images('Train_Classified/Image', classified, step=epoch)
            tl.log_images('Train_Misclassified/Image', misclassified, step=epoch)

        if self.checkpoint_path and self.scheduler.snapshot_before_restart and self.scheduler.needs_checkpoint():
            self.latest_checkpoint = save_checkpoint(self.checkpoint_path, self.config_id, self.budget, self.model,
                                                     self.optimizer, self.scheduler)

        try:
            self.scheduler.step(epoch=epoch)
        except:
            self.scheduler.step(metrics=loss_sum / N, epoch=epoch)

        return [res / N for res in metric_results], loss_sum / N, budget_exceeded, old_proxies

    def evaluate(self, test_loader, metrics, old_proxies, epoch=0):
        N = 0
        metric_results = [0] * len(metrics)

        classified = []
        misclassified = []

        self.model.eval()

        with torch.no_grad():
            for step, (data, targets) in enumerate(test_loader):
                try:
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                except:
                    data = data.to("cpu")
                    targets = targets.to("cpu")

                batch_size = data.size(0)

                outputs = self.model(data)

                # TODO: enable multiple auxiliary outputs
                if isinstance(outputs, tuple):
                    outputs, aux_outputs = outputs

                if self.images_plot_count > 0:
                    _, pred = outputs.topk(1, 1, True, True)
                    pred = pred.t()
                    correct = pred.eq(targets.view(1, -1).expand_as(pred)).cpu().numpy()[0]
                    data = data.cpu().numpy()
                    classified += list(data[correct.astype(bool)])
                    misclassified += list(data[(1 - correct).astype(bool)])
                    if len(classified) > self.images_plot_count:
                        classified = random.sample(classified, self.images_plot_count)
                    if len(misclassified) > self.images_plot_count:
                        misclassified = random.sample(misclassified, self.images_plot_count)

                for i, metric in enumerate(metrics):
                    metric_results[i] += metric(outputs.data, targets.data) * batch_size

                N += batch_size

        if self.images_plot_count > 0:
            import tensorboard_logger as tl
            tl.log_images('Valid_Classified/Image', classified, step=epoch)
            tl.log_images('Valid_Misclassified/Image', misclassified, step=epoch)

        self.model.train()

        if 'val_acc' not in old_proxies.keys():
            old_proxies['val_acc'] = [float(metric_results[0]) / N]
        else:
            old_proxies['val_acc'].append(float(metric_results[0]) / N)

        return [res / N for res in metric_results], old_proxies

    def class_to_probability_mapping(self, test_loader):

        N = 0

        import numpy as np

        probs = None
        class_to_index = dict()
        target_count = []

        self.model.eval()

        with torch.no_grad():
            for i, (data, targets) in enumerate(test_loader):

                data = data.to(self.device)
                targets = targets.to(self.device)

                batch_size = data.size(0)

                try:
                    outputs, aux_outputs = self.model(data)
                except:
                    outputs = self.model(data)

                for i, output in enumerate(outputs):
                    target = targets[i].cpu().item()
                    np_output = output.cpu().numpy()
                    if target not in class_to_index:
                        if probs is None:
                            probs = np.array([np_output])
                        else:
                            probs = np.vstack((probs, np_output))
                        class_to_index[target] = probs.shape[0] - 1
                        target_count.append(0)
                    else:
                        probs[class_to_index[target]] = probs[class_to_index[target]] + np_output

                    target_count[class_to_index[target]] += 1

                N += batch_size

            probs = probs / np.array(target_count)[:, None]
            probs = torch.from_numpy(probs)


        self.model.train()
        return probs, class_to_index
