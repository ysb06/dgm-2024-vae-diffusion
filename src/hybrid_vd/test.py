# Helper script to sample from a conditional DDPM model
# Add project directory to sys.path
import copy
import os

import hydra
import lightning as pl
import torch
import torch.nn as nn
from lightning import seed_everything
from torch.utils.data import DataLoader

# from ...datasets.latent import LatentDataset
from baseline.models.callbacks import ImageWriter
from baseline.models.diffusion import SuperResModel, UNetModel
from baseline.models.diffusion.ddpm import DDPM
from baseline.models.vae import VAE
from baseline.util import configure_device, get_dataset
from hybrid_vd.models.diffuse_vae import DiffuseVAE
from hybrid_vd.models.diffusion.wrapper import DDPMWrapper


def _parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(
    config_path="./configs",
    config_name="test.yaml",
    version_base=None,
)
def sample_cond(config):
    # Seed and setup
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
    model = DDPMWrapper.load_from_checkpoint(
        online_network=online_ddpm_layer,
        target_network=target_ddpm_layer,
        vae=vae_layer,
        **config.ddpm.test_wrapper,
    )
    # 학습 때 use_z를 통해 같이 학습되었어야 하나 그렇게 하지 못함
    # 아래는 임시 코드, 제대로 학습 후 삭제 또는 비활성해야 함

    dataset = get_dataset("cifar10", "./datasets", 32, norm=False, flip=False)
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        persistent_workers=True,
    )

    # Predict trainer
    write_callback = ImageWriter(**config.image_writer)

    trainer = pl.Trainer(**config.trainer, callbacks=[write_callback])
    trainer.predict(model, loader)


if __name__ == "__main__":
    sample_cond()
