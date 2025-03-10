#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import torch
import torch.nn as nn
from .cell_operations import ResNetBasicblock
from .cells import InferCell


# The macro structure for architectures in NAS-Bench-201
class TinyNetwork(nn.Module):

  def __init__(self, C, N, genotype, num_classes, use_bn):
    super(TinyNetwork, self).__init__()
    self._C               = C
    self._layerN          = N
    self.use_bn = use_bn

    if self.use_bn:
      self.stem = nn.Sequential(
                      nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                      nn.BatchNorm2d(C))
    else:
      self.stem = nn.Sequential(
        nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev = C
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2, True)
      else:
        cell = InferCell(genotype, C_prev, C_curr, 1, self.use_bn)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self._Layer = len(self.cells)

    if self.use_bn:
      self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    else:
      self.lastact = nn.ReLU(inplace=True)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)




  def forward_before_global_avg_pool(self, x: torch.Tensor) -> list:
    outputs = []

    def hook_fn(module, inputs, output_t):
      # print(f'Input tensor shape: {inputs[0].shape}')
      # print(f'Output tensor shape: {output_t.shape}')
      outputs.append(inputs[0])

    for m in self.modules():
      if isinstance(m, torch.nn.AdaptiveAvgPool2d):
        m.register_forward_hook(hook_fn)

    self.forward(x)

    assert len(outputs) == 1
    return outputs[0]

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, inputs):
    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits
