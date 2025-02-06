##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
import torch.nn as nn

__all__ = ['OPS', 'ResNetBasicblock', 'SearchSpaceNames']

OPS = {
  'none'         : lambda C_in, C_out, stride, affine, track_running_stats, use_bn: Zero(C_in, C_out, stride),
  'avg_pool_3x3' : lambda C_in, C_out, stride, affine, track_running_stats, use_bn: POOLING(C_in, C_out, stride, 'avg', affine, track_running_stats),
  'nor_conv_3x3' : lambda C_in, C_out, stride, affine, track_running_stats, use_bn: ReLUConvBN(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1), affine, track_running_stats, use_bn),
  'nor_conv_1x1' : lambda C_in, C_out, stride, affine, track_running_stats, use_bn: ReLUConvBN(C_in, C_out, (1,1), (stride,stride), (0,0), (1,1), affine, track_running_stats, use_bn),
  'skip_connect' : lambda C_in, C_out, stride, affine, track_running_stats, use_bn: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}

NAS_BENCH_201         = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

SearchSpaceNames = {'nas-bench-201': NAS_BENCH_201}


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True, use_bn=True):
    super(ReLUConvBN, self).__init__()
    if use_bn:
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not affine),
        nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)
      )
    else:
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not affine)
      )

  def forward(self, x):
    return self.op(x)


class ResNetBasicblock(nn.Module):

  def __init__(self, inplanes, planes, stride, affine=True):
    super(ResNetBasicblock, self).__init__()
    assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
    self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine)
    self.conv_b = ReLUConvBN(  planes, planes, 3,      1, 1, 1, affine)
    if stride == 2:
      self.downsample = nn.Sequential(
                           nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                           nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
    elif inplanes != planes:
      self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine)
    else:
      self.downsample = None
    self.in_dim  = inplanes
    self.out_dim = planes
    self.stride  = stride
    self.num_conv = 2

  def extra_repr(self):
    string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
    return string

  def forward(self, inputs):

    basicblock = self.conv_a(inputs)
    basicblock = self.conv_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(inputs)
    else:
      residual = inputs
    return residual + basicblock


class POOLING(nn.Module):

  def __init__(self, C_in, C_out, stride, mode, affine=True, track_running_stats=True):
    super(POOLING, self).__init__()
    if C_in == C_out:
      self.preprocess = None
    else:
      self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
    if mode == 'avg'  : self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    elif mode == 'max': self.op = nn.MaxPool2d(3, stride=stride, padding=1)
    else              : raise ValueError('Invalid mode={:} in POOLING'.format(mode))

  def forward(self, inputs):
    if self.preprocess: x = self.preprocess(inputs)
    else              : x = inputs
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, C_in, C_out, stride):
    super(Zero, self).__init__()
    self.C_in   = C_in
    self.C_out  = C_out
    self.stride = stride
    self.is_zero = True

  def forward(self, x):
    if self.C_in == self.C_out:
      if self.stride == 1: return x.mul(0.)
      else               : return x[:,:,::self.stride,::self.stride].mul(0.)
    else:
      shape = list(x.shape)
      shape[1] = self.C_out
      zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
      return zeros

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, stride, affine, track_running_stats):
    super(FactorizedReduce, self).__init__()
    self.stride = stride
    self.C_in   = C_in  
    self.C_out  = C_out  
    self.relu   = nn.ReLU(inplace=False)
    if stride == 2:
      #assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
      C_outs = [C_out // 2, C_out - C_out // 2]
      self.convs = nn.ModuleList()
      for i in range(2):
        self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=not affine))
      self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
    elif stride == 1:
      self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
    else:
      raise ValueError('Invalid stride : {:}'.format(stride))
    self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

  def forward(self, x):
    if self.stride == 2:
      x = self.relu(x)
      y = self.pad(x)
      out = torch.cat([self.convs[0](x), self.convs[1](y[:,:,1:,1:])], dim=1)
    else:
      out = self.conv(x)
    out = self.bn(out)
    return out

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)