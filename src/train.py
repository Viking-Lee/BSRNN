import shutil
import typing as tp
import logging
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate                              # 根据配置文件实例化对象
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler

from src.data.dataset import SourceSeparationDataset
from src.data.utils import collate_fn

log = logging.getLogger(__name__)


def initialize_loaders(cfg: DictConfig) -> tp.Tuple[DataLoader, DataLoader]:
    """
    Initializes train and validation dataloaders from configuration file.
    """
    train_dataset = SourceSeparationDataset(**cfg.train_dataset)
    train_loader = DataLoader(train_dataset, **cfg.train_loader, collate_fn=collate_fn)

    if hasattr(cfg, 'val_dataset'):
        val_dataset = SourceSeparationDataset(**cfg.val_dataset)
        val_loader = DataLoader(val_dataset, **cfg.val_loader, collate_fn=collate_fn)
    else:
        val_loader = None
    return (train_loader, val_loader)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg:DictConfig) -> None:
    pl.seed_everything(42, workers=True)
    log.info(OmegaConf.to_yaml(cfg))

    log.info("Initializing loaders, featurizers.")
    train_loader, val_loader = initialize_loaders(cfg)
    # print("train_loader:", train_loader[0])
    # print("val_loader:", val_loader[1])


if __name__ == '__main__':
     my_app()