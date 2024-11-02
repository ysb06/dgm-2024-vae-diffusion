import hydra

@hydra.main(config_path="configs")
def main(config):
    print(type(config))
    print(config)
    print()
    print()
    print(config.pretty())