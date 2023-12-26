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
from src.model.bandsplitrnn import BandSplitRNN

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


def initialize_featurizer(cfg: DictConfig) -> tp.Tuple[nn.Module, nn.Module]:
    """
    Initializes direct and inverse featurizers for audio.
    """
    featurizer = instantiate(cfg.featurizer.direct_transform)
    inv_featurizer = instantiate(cfg.featurizer.inverse_transform)
    return featurizer, inv_featurizer


def initialize_augmentations(cfg: DictConfig) -> nn.Module:
    """
    Initializes augmentations.
    """
    augs = instantiate(cfg.augmentations)
    augs = nn.Sequential(*augs.values())
    return augs


def initialize_model(cfg: DictConfig) -> tp.Tuple[nn.Module, Optimizer, lr_scheduler._LRScheduler]:
    """
    Initializes model from configuration file.
    """
    # initialize model
    model = BandSplitRNN(**cfg.model)
    # initialize optimizer
    if hasattr(cfg, 'opt'):
        opt = instantiate(cfg.opt, params=model.parameters())
    else:
        opt = None
    # initialize scheduler
    if hasattr(cfg, 'sch'):
        if hasattr(cfg.sch, '_target_'):
            # other than LambdaLR
            sch = instantiate(cfg.sch, optimizer=opt)
        else:
            # if LambdaLR
            lr_lambda = lambda epoch: (cfg.sch.alpha ** (cfg.sch.warmup_step - epoch)
                if epoch < cfg.sch.warmup_step
                else cfg.sch.gamma ** (epoch - cfg.sch.warmup_step))
            sch = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lr_lambda)
    else:
        sch = None
    return model, opt, sch


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg:DictConfig) -> None:
    pl.seed_everything(42, workers=True)
    log.info(OmegaConf.to_yaml(cfg))

    log.info("Initializing loaders, featurizers, augmentations.")
    train_loader, val_loader = initialize_loaders(cfg)
    featurizer, inverse_featurizer = initialize_featurizer(cfg)
    augs = initialize_augmentations(cfg)

    log.info("Initializing model, optimizer, scheduler.")
    model, opt, sch = initialize_model(cfg)


if __name__ == '__main__':
     my_app()