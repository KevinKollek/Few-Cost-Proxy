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
import types

from . import proxy
from ..utils.p_utils import get_layer_metric_array
import copy
from torch import nn


@proxy("synflow", bn=False, mode="param")
def compute_synflow_per_weight(net, inputs, targets, bn, mode, split_data=1, loss_fn=None):
    device = inputs.device

    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def network_weight_gaussian_init(net: nn.Module):
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    continue

        return net

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if "weight_mask" not in name:
                param.mul_(signs[name])

    @torch.no_grad()
    def no_op(self, x):
        return x

    @torch.no_grad()
    def copynet(self, bn):
        net = copy.deepcopy(self)
        if bn == False:
            for l in net.modules():
                if isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm1d):
                    l.forward = types.MethodType(no_op, l)
        return net

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

    double_net = copynet(net, bn)

    network_weight_gaussian_init(double_net)

    # keep signs of all params
    signs = linearize(double_net.double())

    # Compute gradients with input of 1s
    double_net.zero_grad()
    double_net.double()
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)

    output = double_net.forward(inputs)
    torch.sum(output[0]).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(double_net, synflow, mode)

    # apply signs of all params
    nonlinearize(double_net, signs)

    grads_abs = sum_mean_arr(grads_abs)

    return grads_abs
