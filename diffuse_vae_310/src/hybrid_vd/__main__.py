import hydra
from omegaconf import DictConfig
import hybrid_vd.train as trainer

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(config: DictConfig):
    trainer.train(config)

main()