import hydra
from omegaconf import DictConfig
import os
from lightning import seed_everything
from baseline.util import get_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from baseline.models.callbacks import EMAWeightUpdate
import lightning as pl
from torch.utils.data import DataLoader

from hybrid_vd.models.diffuse_vae import DiffuseVAE


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig):
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

    model = DiffuseVAE(config)

    config.dataloader.batch_size = min(len(dataset), config.dataloader.batch_size)
    loader = DataLoader(dataset, **config.dataloader)
    
    trainer = pl.Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=loader, ckpt_path=config.ckpt_path)
