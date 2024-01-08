import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig

from train import initialize_model, initialize_featurizer
from utils.utils_inference import load_pl_state_dict, get_minibatch


class Separator(nn.Module):
    """
    which is used in evaluation and inference pipelines
    """
    def __init__(self, cfg: DictConfig, ckpt_path: tp.Optional[str] = None):
        super(Separator, self).__init__()
        self.cfg = cfg

        # modules params
        self.ckpt_path = Path(ckpt_path)


















