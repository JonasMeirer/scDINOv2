from typing import Dict, List, Optional, Tuple, Union, Sequence
from omegaconf import DictConfig
import torch
from torch import Tensor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import math


class MultiViewTransform:
    """Transforms an image into multiple views.

    Args:
        transforms:
            A sequence of transforms. Every transform creates a new view.

    """

    def __init__(self, transforms: Sequence[T.Compose]):
        self.transforms = transforms

    def __call__(self, image: Tensor) -> List[Tensor]:
        """Transforms an image into multiple views.

        Every transform in self.transforms creates a new view.

        Args:
            image:
                Image to be transformed into multiple views.

        Returns:
            List of views.

        """

        if isinstance(image, list):
            # for counterfactual setting, image is a list of two tensors
            trafo_imgs = [self.transforms[0](image[0]), self.transforms[1](image[1])]
            for ind, transform in enumerate(self.transforms[2:]):
                # split local view trafos roughly equally between the two images
                if ind % 2 == 0:
                    trafo_imgs.append(transform(image[0]))
                else:
                    trafo_imgs.append(transform(image[1]))
            return trafo_imgs
        else:
            # for regular setting, image is a single tensor
            return [transform(image) for transform in self.transforms]


class DINOTransform(MultiViewTransform):
    """Implements the global and local view augmentations for DINO [0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2 * global + n_local_views. (8 by default)

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - Random solarization
        - ImageNet normalization

    This class generates two global and a user defined number of local views
    for each image in a batch. The code is adapted from [1].

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes in cofig:
        global_crop_size:
            Crop size of the global views.
        global_crop_scale:
            Tuple of min and max scales relative to global_crop_size.
        local_crop_size:
            Crop size of the local views.
        local_crop_scale:
            Tuple of min and max scales relative to local_crop_size.
        n_local_views:
            Number of generated local views.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        gaussian_blur:
            Tuple of probabilities to apply gaussian blur on the different
            views. The input is ordered as follows:
            (global_view_0, global_view_1, local_views)
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(self, cfg: DictConfig):

        self.cfg = cfg

        global_crop_size = cfg.get("global_crop_size", 50)
        global_crop_scale = cfg.get("global_crop_scale", (0.4, 1.0))
        local_crop_size = cfg.get("local_crop_size", 30)
        local_crop_scale = cfg.get("local_crop_scale", (0.05, 0.4))
        n_local_views = cfg.get("n_local_views", 6)
        hf_prob = cfg.get("hf_prob", 0.5)
        vf_prob = cfg.get("vf_prob", 0)
        rr_prob = cfg.get("rr_prob", 0)
        rr_degrees = cfg.get("rr_degrees", 90)
        intensity_scale_range = cfg.get("intensity_scale_range", (0.5, 1.5))
        intensity_scale_prob = cfg.get("intensity_scale_prob", 0.8)
        intensity_shift_range = cfg.get("intensity_shift_range", (-0.1, 0.1))
        intensity_shift_prob = cfg.get("intensity_shift_prob", 0.5)
        gamma_range = cfg.get("gamma_range", (0.7, 1.5))
        gamma_prob = cfg.get("gamma_prob", 0.5)
        gaussian_noise_sigma = cfg.get("gaussian_noise_sigma", 0.02)
        gaussian_noise_prob = cfg.get("gaussian_noise_prob", 0.8)
        random_channel_drop_prob = cfg.get("random_channel_drop_prob", 0.3)
        cutout_prob = cfg.get("cutout_prob", 0.0)
        cutout_scale = cfg.get("cutout_scale", (0.02, 0.1))
        cutout_ratio = cfg.get("cutout_ratio", (0.3, 3.3))
        gaussian_blur = cfg.get("gaussian_blur", (1.0, 0.1, 0.5))
        kernel_size = cfg.get("kernel_size", 9)
        sigmas = cfg.get("sigmas", (0.1, 2))
        normalize = cfg.get("normalize")

        shared_kwargs = dict(
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            intensity_scale_range=intensity_scale_range,
            intensity_scale_prob=intensity_scale_prob,
            intensity_shift_range=intensity_shift_range,
            intensity_shift_prob=intensity_shift_prob,
            gamma_range=gamma_range,
            gamma_prob=gamma_prob,
            gaussian_noise_sigma=gaussian_noise_sigma,
            gaussian_noise_prob=gaussian_noise_prob,
            random_channel_drop_prob=random_channel_drop_prob,
            cutout_prob=cutout_prob,
            cutout_scale=cutout_scale,
            cutout_ratio=cutout_ratio,
            kernel_size=kernel_size,
            sigmas=sigmas,
            normalize=normalize,
        )

        global_transform_0 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            gaussian_blur=gaussian_blur[0],
            **shared_kwargs,
        )

        global_transform_1 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            gaussian_blur=gaussian_blur[1],
            **shared_kwargs,
        )

        local_transform = DINOViewTransform(
            crop_size=local_crop_size,
            crop_scale=local_crop_scale,
            gaussian_blur=gaussian_blur[2],
            **shared_kwargs,
        )
        local_transforms = [local_transform] * n_local_views
        transforms = [global_transform_0, global_transform_1]
        transforms.extend(local_transforms)
        super().__init__(transforms)


class DINOViewTransform:
    def __init__(
        self,
        crop_size: int = 50,
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        hf_prob: float = 0.5,
        vf_prob: float = 0,
        rr_prob: float = 0,
        rr_degrees: Optional[float] = 90,
        intensity_scale_range: Tuple[float, float] = (0.5, 1.5),
        intensity_scale_prob: float = 0.8,
        intensity_shift_range: Tuple[float, float] = (-0.1, 0.1),
        intensity_shift_prob: float = 0.5,
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        gamma_prob: float = 0.5,
        gaussian_noise_sigma: float = 0.02,
        gaussian_noise_prob: float = 0.8,
        random_channel_drop_prob: float = 0.3,
        cutout_prob: float = 0.0,
        cutout_scale: Tuple[float, float] = (0.02, 0.1),
        cutout_ratio: Tuple[float, float] = (0.3, 3.3),
        gaussian_blur: float = 1.0,
        kernel_size: Optional[float] = 9,
        sigmas: Tuple[float, float] = (0.1, 2),
        normalize: Union[None, Dict[str, List[float]]] = None,
    ):

        transform = []

        if normalize:
            transform.append(T.Normalize(mean=normalize["mean"], std=normalize["std"]))

        transform += [
            T.RandomResizedCrop(
                size=crop_size,
                scale=crop_scale,
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply(
                [T.RandomRotation(degrees=(rr_degrees, rr_degrees))], p=rr_prob
            ),
            T.RandomApply(
                [RandomChannelIntensityScale(scale_range=intensity_scale_range)],
                p=intensity_scale_prob,
            ),
            T.RandomApply(
                [RandomChannelIntensityShift(shift_range=intensity_shift_range)],
                p=intensity_shift_prob,
            ),
            T.RandomApply(
                [RandomChannelGamma(gamma_range=gamma_range)],
                p=gamma_prob,
            ),
            T.RandomApply(
                [AddGaussianNoise(sigma=gaussian_noise_sigma)], p=gaussian_noise_prob
            ),
            T.RandomApply([RandomChannelDrop()], p=random_channel_drop_prob),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=kernel_size, sigma=sigmas)], p=gaussian_blur
            ),
        ]

        if cutout_prob > 0:
            transform.append(
                T.RandomErasing(p=cutout_prob, scale=cutout_scale, ratio=cutout_ratio, value=0)
            )

        self.transform = T.Compose(transform)

    def __call__(self, image: Tensor) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class RandomChannelIntensityScale:
    """Randomly rescales each fluorescence channel independently."""

    def __init__(self, scale_range=(0.5, 1.5)):
        self.scale_range = scale_range

    def __call__(self, x):
        scales = torch.empty(x.size(0)).uniform_(*self.scale_range)
        return x * scales[:, None, None]


class RandomChannelIntensityShift:
    """Adds a random per-channel offset to simulate background fluorescence variation."""

    def __init__(self, shift_range=(-0.1, 0.1)):
        self.shift_range = shift_range

    def __call__(self, x):
        shifts = torch.empty(x.size(0)).uniform_(*self.shift_range)
        return x + shifts[:, None, None]


class RandomChannelGamma:
    """Per-channel gamma correction on non-negative data.

    Applies x_ch^gamma with a different random gamma per channel.
    Negative values are preserved by applying gamma to the absolute value
    and restoring the sign, so this is safe for data centered around zero.
    """

    def __init__(self, gamma_range=(0.7, 1.5)):
        self.log_gamma_lo = math.log(gamma_range[0])
        self.log_gamma_hi = math.log(gamma_range[1])

    def __call__(self, x):
        log_gammas = torch.empty(x.size(0)).uniform_(self.log_gamma_lo, self.log_gamma_hi)
        gammas = log_gammas.exp()[:, None, None]
        return x.sign() * x.abs().pow(gammas)


class AddGaussianNoise:
    """Adds i.i.d. Gaussian noise."""

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, x):
        return x + torch.randn_like(x) * self.sigma


class RandomChannelDrop:
    """Zeros out one randomly chosen fluorescence channel."""

    def __call__(self, x):
        idx = torch.randint(0, x.size(0), (1,))
        out = x.clone()
        out[idx] = 0
        return out
