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
                    grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy()]
        else:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy())
        return grad_dict

    def caculate_zico(grad_dict):
        allgrad_array = None
        for i, modname in enumerate(grad_dict.keys()):
            grad_dict[modname] = np.array(grad_dict[modname])
        nsr_mean_sum = 0
        nsr_mean_sum_abs = 0
        nsr_mean_avg = 0
        nsr_mean_avg_abs = 0
        for j, modname in enumerate(grad_dict.keys()):
            nsr_std = np.std(grad_dict[modname], axis=0)
            nonzero_idx = np.nonzero(nsr_std)[0]
            nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
            tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
            if tmpsum == 0:
                pass
            else:
                nsr_mean_sum_abs += np.log(tmpsum)
                nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]))
        return nsr_mean_sum_abs

    grad_dict = {}
    net.train()

    if torch.cuda.is_available():
        net.cuda()
        data, label = inputs[0].cuda(), targets[0].cuda()
    else:
        data, label = inputs[0], targets[0]
    net.zero_grad()

    logits = net(data)
    loss = loss_fn(logits[1], label)
    loss.backward()
    grad_dict = getgrad(net, grad_dict, 0)

    if torch.cuda.is_available():
        net.cuda()
        data, label = inputs[1].cuda(), targets[1].cuda()
    else:
        data, label = inputs[1], targets[1]
    net.zero_grad()

    logits = net(data)
    loss = loss_fn(logits[1], label)
    loss.backward()
    grad_dict = getgrad(net, grad_dict, 1)

    res = caculate_zico(grad_dict)

    if np.isnan(res) or np.isinf(res):
        res = -100000000

    return float(res)
