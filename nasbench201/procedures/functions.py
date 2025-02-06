#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
import torch
import torch.nn.functional as F
import copy
from nasbench201.utils        import obtain_accuracy
import nasbench201.proxies as proxies

def find_proxy_arrays(net_orig, inputs, targets, proxy_names=None, loss_fn=F.cross_entropy):
    if proxy_names is None:
        proxy_names = proxies.available_proxies

    #move to cpu to free up mem
    torch.cuda.empty_cache()

    done, ds = False, 1
    proxy_values = {}

    while not done:
        try:
            for proxy_name in proxy_names:
                net_copy = copy.deepcopy(net_orig)
                if inputs[0].device == torch.device('cpu'):
                    net_copy = net_copy.cpu()
                else:
                    net_copy = net_copy.cuda()
                if proxy_name not in proxy_values:
                        val = proxies.calc_proxy(proxy_name, net_copy, inputs, targets, loss_fn=loss_fn, split_data=ds)
                        proxy_values[proxy_name] = val

            done = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                done=False
                if ds == inputs[0].shape[0]//2:
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
                  # an array of proxy names to compute, if left blank, all proxies are computed by default
                  proxies_arr=None):  # [not used] if the proxies are already computed but need to be summarized, pass them here

    # Given a neural net
    # and some information about the input data (dataloader)
    # and loss function (loss_fn)
    # this function returns an array of zero-cost proxy metrics.

    def sum_arr(arr):
        sum = 0.
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()

    def sum_mean_arr(arr):
        score = 0
        for grad_abs in arr:
            if len(grad_abs.shape) == 4:
                score += float(torch.mean(torch.sum(grad_abs, dim=[1, 2, 3])))
            elif len(grad_abs.shape) == 2:
                score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
            elif len(grad_abs.shape) == 1:
                score += float(torch.mean(grad_abs))
            else:
                raise RuntimeError('!!!')

        return score

    if proxies_arr is None:
        proxies_arr = find_proxy_arrays(net_orig, inputs, targets, proxy_names, loss_fn=criterion)

    return proxies_arr

def calculate_accuracy(network, data_loader, device):
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
            if device == torch.device('cpu'):
                inputs = inputs.squeeze()
                targets = targets
            else:
                inputs = inputs.squeeze().cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)


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

def obtain_accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

def update_proxies(network, inputs_list, train_loader, targets_list, recursive_loader, proxy_names, criterion, old_proxies, eval_accs=False):
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

    if 'val_acc' in proxy_names:
        val_acc = True
        proxy_names.remove('val_acc')
    else:
        val_acc = False

    proxies = find_proxies(network, inputs_list[-3:], targets_list[-3:], proxy_names=proxy_names, criterion=criterion)

    if val_acc:
        proxy_names.append('val_acc')

    if old_proxies is None:
        # create new dictionary with same keys, but values in lists
        old_proxies = {key: [value] for key, value in proxies.items()}
    else:
        for key, value in proxies.items():
            if key in old_proxies.keys():
                old_proxies[key].append(value)
            else:
                old_proxies[key] = [value]

    if val_acc:
        if 'x-valid' in recursive_loader:
            val_acc = calculate_accuracy(network, recursive_loader['x-valid'])
        else:
            val_acc = calculate_accuracy(network, recursive_loader['ori-test'])

        if 'val_acc' in old_proxies:
            old_proxies['val_acc'].append(val_acc)
        else:
            old_proxies['val_acc'] = [val_acc]

    return old_proxies