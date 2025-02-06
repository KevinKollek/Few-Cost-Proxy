import random
import torch
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def prepare_seed(rand_seed):
  random.seed(rand_seed)
  np.random.seed(rand_seed)
  torch.manual_seed(rand_seed)
  torch.cuda.manual_seed(rand_seed)
  torch.cuda.manual_seed_all(rand_seed)