# Copyright 2024 Kevin Kollek
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
import torch


@proxy("movement", bn=True)
def weight_measures(net, inputs, targets, bn, loss_fn=None, split_data=None):
    weight_measures_dict = {}

    weight_measures_dict['movement'] = []
    for idx, layer in enumerate(net.named_parameters()):
        if len(layer[1].size()) == 4:
            weight_measures_dict['movement'].append((torch.abs(layer[1] - net.param_list[idx])).sum(dim=(0, 1, 2, 3)))
        elif len(layer[1].size()) == 2:
            weight_measures_dict['movement'].append((torch.abs(layer[1] - net.param_list[idx])).sum(dim=(0, 1)))

    weight_measures_dict['movement'] = float(sum(weight_measures_dict['movement']))

    return weight_measures_dict['movement']
