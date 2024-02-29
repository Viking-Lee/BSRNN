import typing as tp

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchaudio.transforms
from torch.optim import Optimizer, lr_scheduler
from omegaconf import DictConfig


class PLModel(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            featurizer: nn.Module,
            inverse_featurizer: nn.Module,
            augmentations: nn.Module,
            opt: Optimizer,
            sch: lr_scheduler._LRScheduler,
            hparams: DictConfig = None
    ):
        super().__init__()

        # augmentations
        self.augmentations = augmentations

        # featurizers
        self.featurizer = featurizer
        self.inverse_featurizer = inverse_featurizer

        # model
        self.model = model

        # losses
        self.mae_specR = nn.L1Loss()
        self.mae_specI = nn.L1Loss()
        self.mae_time = nn.L1Loss()

        # opts
        self.opt = opt
        self.sch = sch

        # logging
        self.save_hyperparameters(hparams)

    def training_step(
            self, batch, batch_idx
    ) -> torch.Tensor:
        """
        Input shape: [batch_size, n_sources, n_channels, time]
        """
        loss, loss_dict, usdr = self.step(batch)

        # logging
        for k in loss_dict:
            self.log(f"train/{k}", loss_dict[k].detach(), on_epoch=True, on_step=False)
        self.log("train/loss", loss.detach(), on_epoch=True, on_step=False)
        self.log("train/usdr", usdr.detach(), on_epoch=True, on_step=False)

        return loss

    def validation_step(
            self, batch, batch_idx
    ) -> torch.Tensor:
        loss, loss_dict, usdr = self.step(batch)
        # logging
        for k in loss_dict:
            self.log(f"val/{k}", loss_dict[k])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/usdr", usdr, prog_bar=True)

        return loss

    def step(
            self, batchT: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor], torch.Tensor]:
        """
        Input shape: [batch_size, n_sources, n_channels, time]
        """
        # augmentations
        batchT = self.augmentations(batchT)

        # STFT
        batchS = self.featurizer(batchT)
        mixS, tgtS = batchS[:, 0], batchS[:, 1]

        # apply model
        predS = self.model(mixS)

        # iSTFT
        batchT = self.inverse_featurizer(
            torch.stack((predS, tgtS), dim=1)
        )
        predT, tgtT = batchT[:, 0], batchT[:, 1]

        # compute loss
        loss, loss_dict = self.compute_multi_resolution_losses(predT, tgtT)

        # compute metrics
        usdr = self.compute_usdr(predT, tgtT)

        return loss, loss_dict, usdr

    def compute_multi_resolution_losses(
        self,
        predT: torch.Tensor,
        tgtT: torch.Tensor,
    ) -> tp.Tuple[torch.Tensro, tp.Dict[str, torch.Tensor]]:
        # multi stft resolution loss
        multi_stft_resolution_loss = 0
        for win_length in [4096, 2048, 1024, 512, 256]:
            transform = torchaudio.transforms.Spectrogram(n_fft = max(win_length, 2048), win_length=win_length, hop_length=147)
            preS = transform(predT)
            tgtS = transform(tgtT)
            multi_stft_resolution_loss = multi_stft_resolution_loss + nn.L1Loss(preS, tgtS)

        # time loss
        time_loss = nn.L1Loss(predT, tgtT)

        loss_dict = {
            "multi_stft_loss": multi_stft_resolution_loss,
            "time_loss": time_loss
        }

        total_loss = time_loss + multi_stft_resolution_loss
        return total_loss, loss_dict

    @staticmethod
    def compute_usdr(
            predT: torch.Tensor,
            tgtT: torch.Tensor,
            delta: float = 1e-7
    ) -> torch.Tensor:
        num = torch.sum(torch.square(tgtT), dim=(1, 2))
        den = torch.sum(torch.square(tgtT - predT), dim=(1, 2))
        num += delta
        den += delta
        usdr = 10 * torch.log10(num / den)
        return usdr.mean()

    def on_before_optimizer_step(
            self, *args, **kwargs
    ):
        norms = pl.utilities.grad_norm(self, norm_type=2)
        norms = dict(filter(lambda elem: '_total' in elem[0], norms.items()))
        self.log_dict(norms)

    def configure_optimizers(self):
        return [self.opt], [self.sch]

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict
