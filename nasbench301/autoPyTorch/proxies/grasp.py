# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from . import proxy
from ..utils.p_utils import get_layer_metric_array


def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()


@proxy("grasp", bn=True, mode="param")
def compute_grasp_per_weight(
        net, inputs, targets, bn, mode, loss_fn, T=1, num_iters=1, split_data=1
):
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True)

    # NOTE original code had some input/target splitting into 2
    # I am guessing this was because of GPU mem limit
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        # forward/grad pass #1
        grad_w = None
        for _ in range(num_iters):
            data, criterion_kwargs = net.loss_computation.prepare_data(inputs[st:en], targets[st:en])
            outputs, aux_outputs = net(data)
            outputs = outputs / T
            aux_outputs = aux_outputs / T

            loss_func = net.loss_computation.criterion(**criterion_kwargs)
            loss = loss_func(net.criterion, outputs)
            loss += 0.4 * loss_func(net.criterion, aux_outputs)

            grad_w_p = autograd.grad(loss, weights, allow_unused=True)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        data, criterion_kwargs = net.loss_computation.prepare_data(inputs[st:en], targets[st:en])

        outputs, aux_outputs = net(data)

        loss_func = net.loss_computation.criterion(**criterion_kwargs)
        loss = loss_func(net.criterion, outputs)

        loss += 0.4 * loss_func(net.criterion, aux_outputs)

        # forward/grad pass #2
        grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)

        # accumulate gradients computed in previous step and call backwards
        z, count = 0, 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    # compute final sensitivity metric and put in grads
    def grasp(layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad  # -theta_q Hg
            # NOTE in the grasp code they take the *bottom* (1-p)% of values
            # but we take the *top* (1-p)%, therefore we remove the -ve sign
            # EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)

    grads = get_layer_metric_array(net, grasp, mode)

    grads = sum_arr(grads)

    if np.isnan(grads) or np.isinf(grads):
        grads = -100000000

    return grads
