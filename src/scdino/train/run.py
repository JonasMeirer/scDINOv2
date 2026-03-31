import hydra
import torch
from omegaconf import DictConfig

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Let's go
    trainer.fit(model=model, datamodule=datamodule)

    # Export teacher backbone in HuggingFace format
    model.save_pretrained("hf_model")


if __name__ == "__main__":
    main()
