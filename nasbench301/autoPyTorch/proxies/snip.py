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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from . import proxy
from ..utils.p_utils import get_layer_metric_array


def snip_forward_conv2d(self, x):
    return F.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()


@proxy("snip", bn=True, mode="param")
def compute_snip_per_weight(net, inputs, targets, bn, mode, loss_fn, split_data=1):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        data, criterion_kwargs = net.loss_computation.prepare_data(inputs[st:en], targets[st:en])

        outputs, aux_outputs = net(data)

        loss_func = net.loss_computation.criterion(**criterion_kwargs)
        loss = loss_func(net.criterion, outputs)

        loss += 0.4 * loss_func(net.criterion, aux_outputs)

        net.zero_grad()
        loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, snip, mode)

    grads_abs = sum_arr(grads_abs)

    if np.isnan(grads_abs) or np.isinf(grads_abs):
        grads_abs = -100000000

    return grads_abs
