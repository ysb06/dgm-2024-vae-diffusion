import hydra
from hydra import initialize, compose
from omegaconf import DictConfig
from omegaconf import DictConfig
import os
from lightning import seed_everything
from baseline.util import get_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from baseline.models.callbacks import EMAWeightUpdate
import lightning as pl
from torch.utils.data import DataLoader
import copy

from hybrid_vd.models.diffuse_vae import DiffuseVAE
from baseline.models.diffusion.ddpm import DDPM
from baseline.models.vae import VAE
from baseline.util import configure_device, get_dataset
from baseline.models.diffusion import SuperResModel, UNetModel
from hybrid_vd.models.diffusion.wrapper import DDPMWrapper


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(config: DictConfig):
    train(config)


def _parse_str(s: str):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


def train(config: DictConfig):
    seed_everything(config.seed)
    dataset = get_dataset(**config.dataset)
    callbacks = []
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(config.results_dir, "checkpoints"),
        filename=f"diffuse_vae" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.chkpt_interval,
        save_on_train_epoch_end=True,
    )
    callbacks.append(chkpt_callback)

    if config.use_ema:
        ema_callback = EMAWeightUpdate(tau=config.ema_decay)
        callbacks.append(ema_callback)

    # Model Definition
    config.ddpm.decoder.attention_resolutions = _parse_str(
        config.ddpm.decoder.attention_resolutions
    )
    config.ddpm.decoder.channel_mult = _parse_str(config.ddpm.decoder.channel_mult)

    vae_layer = VAE(**config.vae.model)
    ddpm_decoder = SuperResModel(**config.ddpm.decoder)
    ddpm_ema_decoder = copy.deepcopy(ddpm_decoder)
    for param in ddpm_ema_decoder.parameters():
        param.requires_grad = False
    online_ddpm_layer = DDPM(ddpm_decoder, **config.ddpm.model)
    target_ddpm_layer = DDPM(ddpm_ema_decoder, **config.ddpm.model)
    model = DDPMWrapper(
        online_ddpm_layer,
        target_ddpm_layer,
        vae_layer,
        **config.ddpm.wrapper,
    )
    # Model Definition End

    config.dataloader.batch_size = min(len(dataset), config.dataloader.batch_size)
    loader = DataLoader(dataset, **config.dataloader)

    trainer = pl.Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=loader, ckpt_path=config.ckpt_path)
