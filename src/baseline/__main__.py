import hydra
from omegaconf import DictConfig

import baseline.train_ae as vae_trainer
import baseline.train_ddpm as ddpm_trainer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig):
    # if config.vae.training.restore_path == "auto":
    # 위 조건에 따른 코드 구현 예정
    # vae_trainer.train(config)
    ddpm_trainer.train(config)


main()
