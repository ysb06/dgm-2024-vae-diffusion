import hydra
from omegaconf import DictConfig
import baseline.train_ae as vae_trainer
import baseline.train_ddpm as ddpm_trainer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig):
    vae_trainer.train(config)
    # config 세팅이 현재 제대로 되어있지 않아서 오류가 발생합니다.
    # ddpm_trainer.train(config)


main()
