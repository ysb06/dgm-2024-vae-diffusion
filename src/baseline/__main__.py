import hydra
from omegaconf import DictConfig
import baseline.train_ae as trainer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig):
    trainer.train(config)


main()
