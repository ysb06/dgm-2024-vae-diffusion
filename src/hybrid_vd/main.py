import hydra
from omegaconf import DictConfig
import os

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig):
    print(type(config))
    print(config)
    print()
    print(os.path.exists(config.test))

main()