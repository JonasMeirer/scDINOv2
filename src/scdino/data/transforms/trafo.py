from omegaconf import DictConfig
from torch import Tensor

from src.scdino.data.transforms.dino import DINOTransform


class Trafo:
    def __init__(self, model: str, mode: str, cfg: DictConfig):

        if mode == "inference":
            # identity transform
            self.transform = lambda x: x
        else:
            if model.name == "dinov2":
                self.transform = DINOTransform(cfg)
            else:
                raise ValueError(
                    f"Invalid combination of model and mode: model:{model}, mode:{mode}"
                )

    def __call__(self, image: Tensor):
        return self.transform(image)
