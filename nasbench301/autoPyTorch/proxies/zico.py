'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
from . import proxy


@proxy("zico", bn=True, mode="param")
def compute_zico_per_weight(net, inputs, targets, bn, mode, split_data=1, loss_fn=None):
    def getgrad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
        if step_iter == 0:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    grad_dict[name] = [mod.weight.grad.detach().clone().cpu().numpy().reshape(-1)]
        else:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    grad_dict[name].append(mod.weight.grad.detach().clone().data.cpu().reshape(-1).numpy())
        return grad_dict

    def caculate_zico(grad_dict):
        allgrad_array = None
        for i, modname in enumerate(grad_dict.keys()):
            grad_dict[modname] = np.array(grad_dict[modname])
        nsr_mean_sum_abs = []
        nsr_mean_avg_abs = []
        for j, modname in enumerate(grad_dict.keys()):
            nsr_std = np.std(grad_dict[modname], axis=0)
            nonzero_idx = np.nonzero(nsr_std)[0]
            nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
            tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
            if tmpsum == 0:
                pass
            else:
                nsr_mean_sum_abs.append(round(float(np.log(tmpsum)), 5))
                nsr_mean_avg_abs.append(
                    round(float(np.log(np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]))), 5))

        return sum(nsr_mean_sum_abs)

    grad_dict = {}
    net.train()

    if inputs[0].device.type == "cuda":
        net.cuda()
        data, label = inputs[0].cuda(), targets[0].cuda()
    elif inputs[0].device.type == "cpu":
        net.cpu()
        data, label = inputs[0].cpu(), targets[0].cpu()

    net.zero_grad()

    data, criterion_kwargs = net.loss_computation.prepare_data(data, label)

    outputs, aux_outputs = net(data)

    loss_func = net.loss_computation.criterion(**criterion_kwargs)
    loss = loss_func(net.criterion, outputs)

    loss += 0.4 * loss_func(net.criterion, aux_outputs)

    loss.backward()

    grad_dict = getgrad(net, grad_dict, 0)

    if inputs[1].device == torch.device(type='cuda'):
        net.cuda()
        data, label = inputs[1].cuda(), targets[1].cuda()
    elif inputs[1].device == torch.device(type='cpu'):
        net.cpu()
        data, label = inputs[1].cpu(), targets[1].cpu()

    net.zero_grad()

    data, criterion_kwargs = net.loss_computation.prepare_data(data, label)

    outputs, aux_outputs = net(data)

    loss_func = net.loss_computation.criterion(**criterion_kwargs)
    loss = loss_func(net.criterion, outputs)

    loss += 0.4 * loss_func(net.criterion, aux_outputs)
    loss.backward()

    grad_dict = getgrad(net, grad_dict, 1)

    for key in grad_dict:
        grad_dict[key] = np.array(grad_dict[key])

    res = caculate_zico(grad_dict)

    if np.isnan(res) or np.isinf(res):
        res = -100000000

    return res
