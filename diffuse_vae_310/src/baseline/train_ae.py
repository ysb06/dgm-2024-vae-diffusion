import logging
import os

import hydra
import lightning as pl
import torchvision.transforms as T
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# from models.vae import VAE
from .models.vae import VAE
from .util import configure_device, get_dataset

logger = logging.getLogger(__name__)


def train(config):
    # Get config and setup
    config = config.dataset.vae
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    dataset = get_dataset(d_type, root, image_size, norm=False, flip=config.data.hflip)
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)

    # Model
    vae = VAE(
        input_res=image_size,
        enc_block_str=config.model.enc_block_config,
        dec_block_str=config.model.dec_block_config,
        enc_channel_str=config.model.enc_channel_config,
        dec_channel_str=config.model.dec_channel_config,
        lr=config.training.lr,
        alpha=config.training.alpha,
    )

    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    # By upgrading from PL 1.4 to 2.0
    # if restore_path is not None:
    #     # Restore checkpoint
    #     train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"vae-{config.training.chkpt_prefix}"
        + "-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    device = config.training.device
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        # By upgrading from PL 1.4 to 2.0
        # train_kwargs["gpus"] = devs
        train_kwargs["devices"] = devs
        train_kwargs["accelerator"] = "gpu"
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    # Fixed for upgrading from PL 1.4 to 2.0
    trainer.fit(vae, train_dataloaders=loader)