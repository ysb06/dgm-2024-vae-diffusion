import hydra
from omegaconf import DictConfig
import baseline.train_ae as vae_trainer
import baseline.train_ddpm as ddpm_trainer
import baseline.test as vae_tester

import torch


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig):
    # vae_trainer.train(config)
    # config 세팅이 현재 제대로 되어있지 않아서 오류가 발생합니다.
    ddpm_trainer.train(config)
    
    # vae_config = config.dataset.vae
    # vae_tester.reconstruct(
    #     vae_config.evaluation.chkpt_path,
    #     vae_config.data.root,
    #     device=vae_config.evaluation.device,
    #     dataset=vae_config.data.name,
    #     image_size=vae_config.data.image_size,
    #     num_samples=vae_config.evaluation.n_samples,
    #     save_path=vae_config.evaluation.save_path,
    #     write_mode=vae_config.evaluation.save_mode,
    # )


main()
