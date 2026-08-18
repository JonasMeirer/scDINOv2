# PROVENANCE (see THIRD_PARTY_NOTICES.md): the losses, masking helpers and
# schedules in this file are vendored from Lightly
# (https://github.com/lightly-ai/lightly), Copyright (c) 2020 Lightly AG and
# its affiliates, used under Lightly's MIT license, reproduced at
# licenses/lightly-MIT.txt. The papers and reference implementations these are
# based on are cited in the individual class docstrings.

import torch
from torch import Tensor
from torch.nn import Module, Parameter, PairwiseDistance, functional
import torch.nn.functional as F
import torch.distributed as dist
import random
import math
import warnings
import numpy as np
from typing import Tuple, Optional, Union


# all utils copied from lightly
# source: https://github.com/lightly-ai/lightly/blob/master/lightly


class DINOLoss(Module):
    """Implementation of the loss described in 'Emerging Properties in
    Self-Supervised Vision Transformers'. [0]

    This implementation follows the code published by the authors. [1]
    It supports global and local image crops. A linear warmup schedule for the
    teacher temperature is implemented to stabilize training at the beginning.
    Centering is applied to the teacher output to avoid model collapse.

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        output_dim:
            Dimension of the model output.
        teacher_temp:
            Temperature parameter for the teacher network.
        student_temp:
            Temperature parameter for the student network.
        center:
            Center used for the teacher output. It is updated with a moving average
            during training.
        center_momentum:
            Momentum term for the center calculation.
        warmup_teacher_temp_epochs:
                Number of epochs for the warmup phase of the teacher temperature (for backward compatibility).
        teacher_temp_schedule:
            A linear schedule for the teacher temperature during the warmup phase (for backward compatibility).

    Examples:
        >>> # initialize loss function
        >>> loss_fn = DINOLoss(128)
        >>>
        >>> # generate a view of the images with a random transform
        >>> view = transform(images)
        >>>
        >>> # embed the view with a student and teacher model
        >>> teacher_out = teacher(view)
        >>> student_out = student(view)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn([teacher_out], [student_out])
    """

    def __init__(
        self,
        output_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        center_mode: str = "mean",
    ) -> None:
        """Initializes the DINOLoss Module.

        Args:
            center_mode:
                Mode for center calculation. Only 'mean' is supported.
            warmup_teacher_temp:
                Initial temperature for the teacher network (for backward compatibility).
            warmup_teacher_temp_epochs:
                Number of epochs for the warmup phase of the teacher temperature (for backward compatibility).
        """
        super().__init__()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

        # TODO(Guarin, 08/24): Refactor this to use the Center module directly once
        # we do a breaking change.
        if center_mode not in CENTER_MODE_TO_FUNCTION:
            raise ValueError(
                f"Unknown mode '{center_mode}'. Valid modes are "
                f"{sorted(CENTER_MODE_TO_FUNCTION.keys())}."
            )
        self._center_fn = CENTER_MODE_TO_FUNCTION[center_mode]
        self.center: Parameter
        self.register_buffer("center", torch.zeros(1, 1, output_dim))
        self.center_momentum = center_momentum

        # comput the warmup teacher temperature internally for backward compatibility
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )

    def forward(
        self,
        teacher_out: list[Tensor],
        student_out: list[Tensor],
        teacher_temp: float | None = None,
        epoch: int | None = None,
    ) -> Tensor:
        """Cross-entropy between softmax outputs of the teacher and student networks.

        Args:
            teacher_out:
                List of tensors with shape (batch_size, output_dim) containing features
                from the teacher model. Each tensor must represent one view of the
                batch.
            student_out:
                List of tensors with shape (batch_size, output_dim) containing features
                from the student model. Each tensor must represent one view of the
                batch.
            teacher_temp:
                The temperature used for the teacher output. If None, the default
                temperature defined in __init__ is used.
            epoch:
                The current epoch for backward compatibility.

        Returns:
            The average cross-entropy loss.
        """

        # Get teacher temperature
        if teacher_temp is not None:
            teacher_temperature = torch.tensor(teacher_temp)
        elif epoch is not None:  # for backward compatibility
            if epoch < self.warmup_teacher_temp_epochs:
                teacher_temperature = self.teacher_temp_schedule[epoch]
            else:
                teacher_temperature = torch.tensor(self.teacher_temp)
        else:
            teacher_temperature = torch.tensor(self.teacher_temp)

        # Calculate cross-entropy loss.
        teacher_out_stacked = torch.stack(teacher_out)
        t_out: Tensor = F.softmax(
            (teacher_out_stacked - self.center) / teacher_temperature, dim=-1
        )
        student_out_stacked = torch.stack(student_out)
        s_out = F.log_softmax(student_out_stacked / self.student_temp, dim=-1)

        # Calculate feature similarities, ignoring the diagonal
        # b = batch_size, t = n_views_teacher, s = n_views_student, d = output_dim
        loss = -torch.einsum("tbd,sbd->ts", t_out, s_out)
        loss.fill_diagonal_(0)

        # Number of loss terms, ignoring the diagonal
        n_terms = loss.numel() - loss.diagonal().numel()
        batch_size = teacher_out_stacked.shape[1]

        # scDINO deviation from upstream Lightly: guard the degenerate case.
        # With a single teacher and a single student view every pair lies on the
        # excluded diagonal, so n_terms is 0 and the division below silently
        # produces NaN. Fail loudly instead of poisoning the loss.
        if n_terms == 0:
            raise ValueError(
                "DINOLoss received no off-diagonal view pairs "
                f"({len(teacher_out)} teacher view(s), {len(student_out)} student "
                "view(s)). Same-index view pairs are excluded, so at least one "
                "side needs a second view."
            )

        loss = loss.sum() / (n_terms * batch_size)

        # Update the center used for the teacher output
        self.update_center(teacher_out_stacked)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_out: Tensor) -> None:
        """Moving average update of the center used for the teacher output.

        Args:
            teacher_out:
                Tensor with shape (num_views, batch_size, output_dim) containing
                features from the teacher model.
        """

        # Calculate the batch center using the specified center function
        batch_center = self._center_fn(x=teacher_out, dim=(0, 1))

        # Update the center with a moving average
        self.center.data = center_momentum(
            center=self.center, batch_center=batch_center, momentum=self.center_momentum
        )


class Center(Module):
    """Center module to compute and store the center of a feature tensor as used
    in DINO [0].

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294

    Attributes:
        size:
            Size of the tracked center tensor. Dimensions across which the center
            is computed must be set to 1. For example, if the feature tensor has shape
            (batch_size, sequence_length, feature_dim) and the center should be computed
            across the batch and sequence dimensions, the size should be
            (1, 1, feature_dim).
        mode:
            Mode to compute the center. Currently only 'mean' is supported.
        momentum:
            Momentum term for the center calculation.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        mode: str = "mean",
        momentum: float = 0.9,
    ) -> None:
        """Initializes the Center module with the specified parameters.

        Raises:
            ValueError: If an unknown mode is provided.
        """
        super().__init__()

        center_fn = CENTER_MODE_TO_FUNCTION.get(mode)
        if center_fn is None:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes are "
                f"{sorted(CENTER_MODE_TO_FUNCTION.keys())}."
            )
        self._center_fn = center_fn

        self.size = size
        self.dim = tuple(i for i, s in enumerate(size) if s == 1)
        self.center: Tensor  # For mypy
        self.register_buffer("center", torch.zeros(self.size))
        self.momentum = momentum

    @property
    def value(self) -> Tensor:
        """The current value of the center.

        Use this property to do any operations based on the center.
        """
        return self.center

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Update the center with a new batch of features.

        Args:
            x:
                Feature tensor used to update the center. Must have the same number of
                dimensions as self.size.
        """
        batch_center = self._center_fn(x=x, dim=self.dim)
        self.center = center_momentum(
            center=self.center, batch_center=batch_center, momentum=self.momentum
        )

    @torch.no_grad()
    def _center_mean(self, x: Tensor) -> Tensor:
        """Returns the center of the input tensor by calculating the mean."""
        return center_mean(x=x, dim=self.dim)


@torch.no_grad()
def center_mean(x: Tensor, dim: Tuple[int, ...]) -> Tensor:
    """Returns the center of the input tensor by calculating the mean.

    Args:
        x:
            Input tensor.
        dim:
            Dimensions along which the mean is calculated.

    Returns:
        The center of the input tensor.
    """
    batch_center = torch.mean(x, dim=dim, keepdim=True)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(batch_center)
        batch_center = batch_center / dist.get_world_size()
    return batch_center


@torch.no_grad()
def center_momentum(center: Tensor, batch_center: Tensor, momentum: float) -> Tensor:
    """Returns the new center with momentum update."""
    return center * momentum + batch_center * (1 - momentum)


CENTER_MODE_TO_FUNCTION = {
    "mean": center_mean,
}


class IBOTPatchLoss(Module):
    """Implementation of the iBOT patch loss [0] as used in DINOv2 [1].

    Implementation is based on [2].

    - [0]: iBOT, 2021, https://arxiv.org/abs/2111.07832
    - [1]: DINOv2, 2023, https://arxiv.org/abs/2304.07193
    - [2]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py

    Attributes:
        output_dim:
            Dimension of the model output.
        teacher_temp:
            Temperature for the teacher output.
        student_temp:
            Temperature for the student output.
        center_mode:
            Mode for center calculation. Only 'mean' is supported.
        center_momentum:
            Momentum term for the center update.
    """

    def __init__(
        self,
        output_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_mode: str = "mean",
        center_momentum: float = 0.9,
    ) -> None:
        """Initializes the iBOTPatchLoss module with the specified parameters."""
        super().__init__()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

        self.center = Center(
            size=(1, output_dim),
            mode=center_mode,
            momentum=center_momentum,
        )

    def forward(
        self,
        teacher_out: Tensor,
        student_out: Tensor,
        mask: Tensor,
        teacher_temp: float | None = None,
    ) -> Tensor:
        """Forward pass through the iBOT patch loss.

        Args:
            teacher_out:
                Tensor with shape (batch_size * sequence_length, embed_dim) containing
                the teacher output of the masked tokens.
            student_out:
                Tensor with shape (batch_size * sequence_length, embed_dim) containing
                the student output of the masked tokens.
            mask:
                Boolean tensor with shape (batch_size, height, width) containing the
                token mask. Exactly batch_size * sequence_length entries must be set to
                True in the mask.
            teacher_temp:
                The temperature used for the teacher output. If None, the default
                temperature defined in __init__ is used.

        Returns:
            The loss value.
        """
        # B = batch size, N = sequence length = number of masked tokens, D = embed dim
        # H = height (in tokens), W = width (in tokens)
        # Note that N <= H * W depending on how many tokens are masked.
        teacher_temperature = torch.tensor(
            teacher_temp if teacher_temp is not None else self.teacher_temp
        )

        # Calculate cross-entropy loss.
        teacher_softmax = F.softmax(
            (teacher_out - self.center.value) / teacher_temperature, dim=-1
        )
        student_log_softmax = F.log_softmax(student_out / self.student_temp, dim=-1)

        # (B * N, D) -> (B * N)
        loss = -torch.sum(teacher_softmax * student_log_softmax, dim=-1)

        # Get weights.
        # (B, H, W) -> (B, 1, 1)
        num_masked_per_image = mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        # (B, 1, 1) -> (B, H, W) -> (B * N)
        weight = (1.0 / num_masked_per_image).expand_as(mask)[mask]

        # Apply weighting.
        B = mask.shape[0]
        loss = (loss * weight).sum() / B

        self.center.update(teacher_out)

        return loss


class KoLeoLoss(Module):
    """KoLeo loss based on [0].

    KoLeo loss is a regularizer that encourages a uniform span of the features in a
    batch by penalizing the distance between the features and their nearest
    neighbors.

    Implementation is based on [1].

    - [0]: Spreading vectors for similarity search, 2019, https://arxiv.org/abs/1806.03198
    - [1]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py

    Attributes:
        p:
            The norm degree for pairwise distance calculation.
        eps:
            Small value to avoid division by zero.
    """

    def __init__(
        self,
        p: float = 2,
        eps: float = 1e-8,
    ):
        """Initializes the KoLeoLoss module with the specified parameters.

        Args:
            p:
                The norm degree for pairwise distance calculation.
            eps:
                Small value to avoid division by zero.
        """

        super().__init__()
        self.p = p
        self.eps = eps
        self.pairwise_distance = PairwiseDistance(p=p, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through KoLeo Loss.

        Args:
            x: Tensor with shape (batch_size, embedding_size).

        Returns:
            Loss value.
        """
        # Normalize the input tensor
        x = functional.normalize(x, p=2, dim=-1, eps=self.eps)

        # Calculate cosine similarity.
        cos_sim = torch.mm(x, x.t())
        cos_sim.fill_diagonal_(-2)

        # Get nearest neighbors.
        nn_idx = cos_sim.argmax(dim=1)
        nn_dist: Tensor = self.pairwise_distance(x, x[nn_idx])

        # Compute the loss
        loss = -(nn_dist + self.eps).log().mean()

        return loss


# class SIGRegLoss(Module):
#     def __init__(
#         self,
#         knots: int = 17,
#         num_slices: int = 256,
#         max_t: float = 3.0,
#     ):
#         super().__init__()

#         t = torch.linspace(0, max_t, knots, dtype=torch.float32)
#         dt = max_t / (knots - 1)

#         weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
#         weights[[0, -1]] = dt

#         phi = torch.exp(-t.square() / 2.0)

#         self.num_slices = num_slices
#         self.register_buffer("t", t)
#         self.register_buffer("phi", phi)
#         self.register_buffer("weights", weights * phi)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor with shape [batch_size, embedding_dim]
#         """
#         A = torch.randn(
#             x.size(-1),
#             self.num_slices,
#             device=x.device,
#             dtype=x.dtype,
#         )
#         A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

#         t = self.t.to(dtype=x.dtype)
#         phi = self.phi.to(dtype=x.dtype)
#         weights = self.weights.to(dtype=x.dtype)

#         x_t = (x @ A).unsqueeze(-1) * t

#         real_err = (x_t.cos().mean(dim=0) - phi).square()
#         imag_err = x_t.sin().mean(dim=0).square()

#         err = real_err + imag_err

#         statistic = (err @ weights) * x.size(0)

#         return statistic.mean()


def random_block_mask(
    size: Tuple[int, int, int],
    batch_mask_ratio: float = 0.5,
    min_image_mask_ratio: float = 0.1,
    max_image_mask_ratio: float = 0.5,
    min_num_masks_per_block: int = 4,
    max_num_masks_per_block: Optional[int] = None,
    min_block_aspect_ratio: float = 0.3,
    max_block_aspect_ratio: Optional[float] = None,
    max_attempts_per_block: int = 10,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Creates a random block mask for a batch of images.

    A block is in this context a rectangle of patches in an image that are
    masked together. The function generates block masks until the desired number of
    patches per image are masked. DINOv2 uses a more complex masking strategy that
    only generates masks for mask_ratio of the images. On top of that, it also masks
    a different number of patches for every image. This is controlled by the
    min_image_mask_ratio and max_image_mask_ratio arguments.

    Based on the implementation of the block mask in DINOv2 [0]. For details see [1]
    and [2].

    - [0]: DINOv2, 2023, https://arxiv.org/abs/2304.07193
    - [1]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/masking.py
    - [2]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/collate.py

    Args:
        size:
            Size of the image batch for which to generate masks.
            Should be (batch_size, height, width).
        batch_mask_ratio:
            Percentage of images per batch for which to generate block masks.
            The remaining images are not masked.
        min_image_mask_ratio:
            Minimum percentage of the image to mask. In practice, fewer than
            min_image_mask_ratio patches of the image can be masked due to additional
            constraints.
        max_image_mask_ratio:
            Maximum percentage of the image to mask.
        min_num_masks_per_block:
            Minimum number of patches to mask per block.
        max_num_masks_per_block:
            Maximum number of patches to mask per block.
        min_block_aspect_ratio:
            Minimum aspect ratio (height/width) of a masked block.
        max_block_aspect_ratio:
            Maximum aspect ratio (height/width) of a masked block.
        max_attempts_per_block:
            Maximum number of attempts to find a valid block mask for an image.
        device:
            Device on which to create the mask.
    Returns:
        A boolean tensor with shape (batch_size, height, width) where each entry
        is True if the patch should be masked and False otherwise.

    Raises:
        ValueError: If 'max_image_mask_ratio' is less than 'min_image_mask_ratio'.
    """

    if max_image_mask_ratio < min_image_mask_ratio:
        raise ValueError(
            "max_image_mask_ratio must be greater or equal to min_image_mask_ratio."
        )

    # B is batch size(number of images), H is height, W is width
    B, H, W = size
    num_images_masked = int(B * batch_mask_ratio)
    probs = torch.linspace(
        min_image_mask_ratio, max_image_mask_ratio, num_images_masked + 1
    ).tolist()
    image_masks = []
    for prob_min, prob_max in zip(probs[:-1], probs[1:]):
        num_mask = int(H * W * random.uniform(prob_min, prob_max))
        num_mask = max(num_mask, min_num_masks_per_block)
        image_masks.append(
            random_block_mask_image(
                size=(H, W),
                num_masks=num_mask,
                min_num_masks_per_block=min_num_masks_per_block,
                max_num_masks_per_block=max_num_masks_per_block,
                min_block_aspect_ratio=min_block_aspect_ratio,
                max_block_aspect_ratio=max_block_aspect_ratio,
                max_attempts_per_block=max_attempts_per_block,
                device=device,
            )
        )

    # Add non-masked images to fill the batch
    for _ in range(num_images_masked, B):
        image_masks.append(torch.zeros((H, W), dtype=torch.bool, device=device))

    random.shuffle(image_masks)
    return torch.stack(image_masks)


def random_block_mask_image(
    size: Tuple[int, int],
    num_masks: int,
    min_num_masks_per_block: int = 4,
    max_num_masks_per_block: Optional[int] = None,
    min_block_aspect_ratio: float = 0.3,
    max_block_aspect_ratio: Optional[float] = None,
    max_attempts_per_block: int = 10,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Creates a random block mask for a single image.

    Args:
        size:
            Size of the image for which to generate a mask.
            Should be (height, width).
        num_masks:
            Number of patches to mask.
        min_num_masks_per_block:
            Minimum number of patches to mask per block.
        max_num_masks_per_block:
            Maximum number of patches to mask per block.
        min_block_aspect_ratio:
            Minimum aspect ratio (height/width) of a masked block.
        max_block_aspect_ratio:
            Maximum aspect ratio (height/width) of a masked block.
        max_attempts_per_block:
            Maximum number of attempts to find a valid block mask.
        device:
            Device on which to create the mask.
    Returns:
        A boolean tensor with shape (height, width) where each entry is True if the
        patch should be masked and False otherwise.

    Raises:
        ValueError: If 'max_num_masks_per_block' is less than 'min_num_masks_per_block' or
            if 'max_block_aspect_ratio' is less than 'min_block_aspect_ratio'
    """

    if max_block_aspect_ratio is None:
        max_block_aspect_ratio = 1 / min_block_aspect_ratio
    if max_num_masks_per_block is None:
        max_num_masks_per_block = num_masks

    if max_num_masks_per_block < min_num_masks_per_block:
        raise ValueError(
            "max_num_masks_per_block must be greater or equal to min_num_masks_per_block."
        )
    if max_block_aspect_ratio < min_block_aspect_ratio:
        raise ValueError(
            "max_block_aspect_ratio must be greater or equal to min_block_aspect_ratio."
        )

    log_min_aspect = math.log(min_block_aspect_ratio)
    log_max_aspect = math.log(max_block_aspect_ratio)

    H, W = size
    mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    mask_count = 0
    while mask_count < num_masks:
        # Try masking a block
        max_new_masked = min(num_masks - mask_count, max_num_masks_per_block)
        delta = 0
        for _ in range(max_attempts_per_block):
            target_area = random.uniform(min_num_masks_per_block, max_new_masked)
            aspect_ratio = math.exp(random.uniform(log_min_aspect, log_max_aspect))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                num_already_masked = mask[top : top + h, left : left + w].sum().item()
                num_new_masked = h * w - num_already_masked
                if 0 < num_new_masked <= max_new_masked:
                    mask[top : top + h, left : left + w] = 1
                    delta += num_new_masked
            if delta > 0:
                break
        if delta == 0:
            break
        else:
            mask_count += delta
    return mask


@torch.no_grad()
def update_momentum(model: Module, model_ema: Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`

    Momentum encoders are a crucial component for models such as MoCo or BYOL.

    Args:
        model:
            The current model.
        model_ema:
            The model with exponential moving average (EMA) parameters.
        m:
            The momentum factor, between 0 and 1.

    Examples:
        >>> backbone = resnet18()
        >>> projection_head = MoCoProjectionHead()
        >>> backbone_momentum = copy.deepcopy(moco)
        >>> projection_head_momentum = copy.deepcopy(projection_head)
        >>>
        >>> # update momentum
        >>> update_momentum(moco, moco_momentum, m=0.999)
        >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
    """
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


def cosine_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    period: Optional[int] = None,
) -> float:
    """Use cosine decay to gradually modify start_value to reach target end_value.

    Args:
        step:
            Current step number.
        max_steps:
            Total number of steps.
        start_value:
            Starting value.
        end_value:
            Target value.
        period:
            The number of steps over which the cosine function completes a full cycle.
            If no period is provided, the scheduler will complete a half cycle over
            max_steps.

    Returns:
        Cosine decay value.

    """
    if step < 0:
        raise ValueError(f"Current step number {step} can't be negative.")
    if max_steps < 0:
        raise ValueError(f"Total step number {max_steps} can't be negative.")
    if period is None and step > max_steps:
        warnings.warn(
            f"Current step number {step} exceeds max_steps {max_steps}.",
            category=RuntimeWarning,
        )
    if period is not None and period <= 0:
        raise ValueError(f"Period {period} must be >= 1")

    decay: float
    if period is not None:  # "cycle" based on period, if provided
        decay = (
            end_value
            - (end_value - start_value) * (np.cos(2 * np.pi * step / period) + 1) / 2
        )
    elif max_steps <= 1:
        # Avoid division by zero
        decay = end_value
    elif step >= max_steps - 1:
        # Special case for Pytorch Lightning which updates LR scheduler also for epoch
        # after last training epoch.
        decay = end_value
    else:
        decay = (
            end_value
            - (end_value - start_value)
            * (np.cos(np.pi * step / (max_steps - 1)) + 1)
            / 2
        )
    return decay


def linear_warmup_schedule(
    step: int,
    warmup_steps: int,
    start_value: float,
    end_value: float,
) -> float:
    if warmup_steps < 0:
        raise ValueError(f"Warmup steps {warmup_steps} can't be negative.")
    if step < 0:
        raise ValueError(f"Current step number {step} can't be negative.")
    if start_value < 0:
        raise ValueError(f"Start value {start_value} can't be negative.")
    if end_value <= 0:
        raise ValueError(f"End value {end_value} can't be non-positive.")
    if start_value > end_value:
        raise ValueError(
            f"Start value {start_value} must be less than or equal to end value {end_value}."
        )
    if step < warmup_steps:
        return start_value + step / warmup_steps * (end_value - start_value)
    else:
        return end_value
