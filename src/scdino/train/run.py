import json
import time
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)

    if cfg.eval_only_at_end:
        trainer.limit_val_batches = 0

    start_time = time.time()
    trainer.fit(model=model, datamodule=datamodule)
    trainer.limit_val_batches = 1.0
    trainer.validate(model=model, datamodule=datamodule)
    wall_time = time.time() - start_time

    # Export teacher backbone in HuggingFace format
    model.save_pretrained("hf_model")

    # Write benchmark results
    num_params = sum(p.numel() for p in model.parameters())
    train_samples = len(datamodule.train_dataset)

    metrics = {}
    for key in ["val/knn_top1", "val/knn_top5", "val/silhouette"]:
        val = trainer.callback_metrics.get(key)
        metrics[key.replace("/", "_")] = (
            format(float(val), ".4f") if val is not None else None
        )

    results = {
        "seed": cfg.seed,
        "metrics": metrics,
        "model": {"num_parameters": num_params},
        "data": {"num_train_samples": train_samples},
        "runtime": {
            "wall_time_seconds": round(wall_time, 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "num_gpus": torch.cuda.device_count(),
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    Path("results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
