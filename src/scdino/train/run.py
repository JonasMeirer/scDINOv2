import torch
torch.set_float32_matmul_precision("medium")
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Let's go
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()