##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################

# useful modules
from nasbench201.configs.config_utils import dict2config
from .tiny_network import TinyNetwork
from .genotypes import Structure as CellStructure

# Cell-based NAS Models
def get_cell_based_tiny_net(config):
  if isinstance(config, dict): config = dict2config(config, None) # to support the argument being a dict
  if hasattr(config, 'genotype'):
    genotype = config.genotype
  return TinyNetwork(config.C, config.N, genotype, config.num_classes, config.use_bn)