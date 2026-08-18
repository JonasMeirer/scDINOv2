from omegaconf import DictConfig
from torch import Tensor

from scdino.data.transforms.dino import DINOTransform


class Trafo:
    def __init__(self, model: str, mode: str, cfg: DictConfig):

        if mode == "inference":
            # no transform
            self.transform = lambda x: x
        else:
            if model.name in ["dinov2", "dino", "dinov3", "dinov2_StrucPerc"]:
                self.transform = DINOTransform(cfg)
            else:
                raise ValueError(
                    f"Invalid combination of model: {model.name} and mode:{mode}"
                )

    def __call__(self, image: Tensor):
        return self.transform(image)
