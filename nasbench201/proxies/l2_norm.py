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

from . import proxy
from . import p_utils
import torch
import numpy as np


def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()


@proxy("l2_norm", copy_net=False, mode="param")
def get_l2_norm_array(net, inputs, targets, bn, mode, split_data=1, **kwargs):
    metric = p_utils.get_layer_metric_array(net, lambda l: l.weight.norm(), mode=mode)

    metric = sum_arr(metric)

    if np.isnan(metric) or np.isinf(metric):
        metric = -100000000

    return metric
