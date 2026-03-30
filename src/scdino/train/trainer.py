import torch
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig


class HydraTrainer(L.Trainer):
    def __init__(
        self,
        max_epochs: int,
        devices: int | str | list[int],
        hardware_accelerator: str,
        check_val_every_n_epoch: int = 1,
        log_every_n_steps: int = 50,
        limit_train_batches: int | float | None = None,
        ckpt_every_n_epochs: int | None = None,
        default_root_dir: str | None = None,
        logging: DictConfig | None = None,
    ) -> None:
        accelerator = hardware_accelerator if torch.cuda.is_available() else "cpu"
        logger = self._build_logger(logging)
        callbacks = self._build_callbacks(ckpt_every_n_epochs, default_root_dir)

        trainer_kwargs = {
            "max_epochs": max_epochs,
            "devices": devices,
            "accelerator": accelerator,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "log_every_n_steps": log_every_n_steps,
            "limit_train_batches": limit_train_batches,
            "default_root_dir": default_root_dir,
        }

        if logger is not None:
            trainer_kwargs["logger"] = logger
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        super().__init__(**trainer_kwargs)

    @staticmethod
    def _build_logger(logging_cfg: DictConfig | None):
        if logging_cfg is None:
            return None

        if logging_cfg.get("name") == "mlflow":
            from lightning.pytorch.loggers import MLFlowLogger
            return MLFlowLogger(
                experiment_name=logging_cfg.get("experiment_name"),
                tracking_uri=logging_cfg.get("tracking_uri"),
                tags=logging_cfg.get("tags"),
                log_model=logging_cfg.get("log_model"),
            )
        elif logging_cfg.get("name") == "wandb":
            from lightning.pytorch.loggers import WandbLogger
            return WandbLogger(
                project=logging_cfg.get("project"),
                entity=logging_cfg.get("entity"),
                tags=logging_cfg.get("tags"),
            )
        elif logging_cfg.get("name") == "console":
            from lightning.pytorch.loggers import CSVLogger
            return CSVLogger(
                save_dir=logging_cfg.get("save_dir", "logs"),
                name=logging_cfg.get("logger_name", "console"),
            )
        else:
            raise ValueError(
                f"Invalid logging configuration: {logging_cfg.get('name')}"
            )

    @staticmethod
    def _build_callbacks(
        ckpt_every_n_epochs: int | None, default_root_dir: str | None
    ) -> list:
        if not ckpt_every_n_epochs:
            return []

        from lightning.pytorch.callbacks import ModelCheckpoint

        return [
            ModelCheckpoint(
                dirpath=default_root_dir,
                filename="{epoch:02d}-{val_knn_top1:.2f}",
                save_top_k=-1,
                every_n_epochs=ckpt_every_n_epochs,
            )
        ]