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

available_proxies = []
_proxy_impls = {}


def proxy(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        def proxy_impl(net_orig, inputs, targets, *args, **kwargs):
            import torch
            if inputs[0].device.type == "cuda":
                net = net_orig.cuda()
                inputs[0] = inputs[0].cuda()
            elif inputs[0].device.type == "cpu":
                net = net_orig.cpu()
                inputs[0] = inputs[0].cpu()

            net.device = inputs[0].device

            ret = func(net, inputs, targets, bn, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc

                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        global _proxy_impls
        if name in _proxy_impls:
            raise KeyError(f"Duplicated proxy! {name}")
        available_proxies.append(name)
        _proxy_impls[name] = proxy_impl
        return func

    return make_impl


def calc_proxies(name, net, inputs, targets, *args, **kwargs):
    if name == 'LinReg' or name == 'zico':
        return _proxy_impls[name](net, inputs, targets, *args, **kwargs)
    else:
        return _proxy_impls[name](net, inputs[0], targets[0], *args, **kwargs)


def load_all():
    from . import grad_norm
    from . import snip
    from . import grasp
    from . import fisher
    from . import jacov
    from . import plain
    from . import synflow
    from . import epe_nas
    from . import zen
    from . import l2_norm
    from . import nwot
    from . import large_final
    from . import magnitude_increase
    from . import movement
    from . import absmaginc
    from . import zico


# TODO: should we do that by default?
load_all()
