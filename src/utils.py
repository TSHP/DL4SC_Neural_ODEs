import random
import yaml
from pathlib import Path

import numpy as np
import torch


# Load yaml config file
def load_config(path: Path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def kl_div(mu, log_var):
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)


