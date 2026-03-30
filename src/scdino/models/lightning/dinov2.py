import torch
from torch import Tensor
from torch.nn import Module, Parameter, PairwiseDistance, functional
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Any, Optional, Union

import warnings
import random
import math
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from typing import Tuple

import lightning as L

from src.scdino.models.backbones.dinov2 import DINOv2 as DINOv2Skeleton
from src.scdino.eval.knn import knn_classifier, compute_knn_accuracy


class DINOv2(L.LightningModule):
    def __init__(
        self,
        name: str,
        architecture: Dict[str, Any],
        training: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Store configs
        self.name = name
        self.backbone_config = architecture.backbone
        self.dino_head_config = architecture.dino_head
        self.ibot_head_config = architecture.ibot_head
        self.ibot_separate_head = architecture.ibot_separate_head
        self.knn_eval_config = training.knn_eval
        self.training_config = training
        
        model = DINOv2Skeleton(
            backbone_config=self.backbone_config,
            dino_head_config=self.dino_head_config,
            ibot_head_config=self.ibot_head_config,
            ibot_separate_head=self.ibot_separate_head
        )

        self.teacher_backbone = model.teacher_backbone
        self.student_backbone = model.student_backbone

        self.teacher_head = model.teacher_head
        self.student_head = model.student_head

        # Losses
        dino_loss_config = self.training_config["losses"]["dino"]
        ibot_loss_config = self.training_config["losses"]["ibot"]
        koleo_loss_config = self.training_config["losses"]["koleo"]
        
        self.dino_criterion = DINOLoss(
            output_dim=dino_loss_config["output_dim"],
            warmup_teacher_temp=dino_loss_config["warmup_teacher_temp"],
            teacher_temp=dino_loss_config["teacher_temp"],
            warmup_teacher_temp_epochs=dino_loss_config["warmup_teacher_temp_epochs"],
            student_temp=dino_loss_config["student_temp"],
            center_momentum=dino_loss_config["center_momentum"],
            center_mode=dino_loss_config["center_mode"]
        )
        self.ibot_criterion = IBOTPatchLoss(
            output_dim=ibot_loss_config["output_dim"],
            teacher_temp=ibot_loss_config["teacher_temp"],
            student_temp=ibot_loss_config["student_temp"],
            center_mode=ibot_loss_config["center_mode"],
            center_momentum=ibot_loss_config["center_momentum"]
        )
        self.koleo_criterion = KoLeoLoss(
            p=koleo_loss_config["p"],
            eps=koleo_loss_config["eps"]
        )
        
        # kNN evaluation configuration
        self.enable_knn_eval = self.knn_eval_config.enable_knn_eval
        self.knn_k = self.knn_eval_config.knn_k
        self.knn_temperature = self.knn_eval_config.knn_temperature
        self.knn_max_train_batches = self.knn_eval_config.knn_max_train_batches
        self.knn_val_chunk_size = self.knn_eval_config.knn_val_chunk_size
        self.knn_train_chunk_size = self.knn_eval_config.knn_train_chunk_size
        
        # Storage for training and validation features and labels (only if kNN eval is enabled)
        if self.enable_knn_eval:
            self.train_features = []
            self.train_labels = []
            self.validation_features = []
            self.validation_labels = []

    def forward(self, x: Tensor) -> Tensor:
        pass
    
    def encode(self, x: Tensor) -> Tensor:
        """ For compatibility in the embed.py script """
        self.teacher_backbone.eval()
        with torch.no_grad():
            embeds = self.teacher_backbone.encode(x)[:,0]
        self.teacher_backbone.train()
        return embeds

    def forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(
        self, x: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features

    def training_step(
        self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        views, _ = batch[0], batch[1]
        global_views = torch.cat(views[:2])
        local_views = torch.cat(views[2:])

        # Masking
        B = len(global_views)
        sequence_length = self.teacher_backbone.sequence_length
        mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)
        # Mask patches except class token.
        H, W = self.teacher_backbone.vit.patch_embed.grid_size
        n_registered_tokens = self.teacher_backbone.vit.num_prefix_tokens
        assert (
            H * W == sequence_length - n_registered_tokens
        ), f"Unexpected grid size: {H}x{W}, sequence_length {sequence_length}"
        block_mask = random_block_mask(size=(B, H, W), device=mask.device)
        mask[:, n_registered_tokens:] = block_mask.flatten(start_dim=1)

        # Teacher forward
        with torch.no_grad():
            teacher_cls_token, teacher_features = self.forward_teacher(global_views)
            teacher_cls_out = self.teacher_head.dino_head.forward(teacher_cls_token)
            teacher_masked_out = self.teacher_head.ibot_head.forward(
                teacher_features[mask]
            )

        # Student forward
        student_global_cls_token, student_global_masked_features = self.forward_student(
            global_views, mask=mask
        )
        student_global_cls_out = self.student_head.dino_head.forward(
            student_global_cls_token
        )
        student_global_masked_out = self.student_head.ibot_head.forward(
            student_global_masked_features
        )

        student_local_cls_token, _ = self.forward_student(local_views, mask=None)
        student_local_cls_out = self.student_head.dino_head.forward(
            student_local_cls_token
        )
        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])

        teacher_temp_config = self.training_config["teacher_temp"]
        teacher_temp = linear_warmup_schedule(
            step=self.trainer.global_step,
            warmup_steps=int(
                teacher_temp_config["warmup_steps"] / self.trainer.max_epochs * self.trainer.estimated_stepping_batches
            ),
            start_value=teacher_temp_config["start_value"],
            end_value=teacher_temp_config["end_value"],
        )
        dino_loss = self.dino_criterion(
            teacher_out=teacher_cls_out.chunk(2),
            student_out=student_cls_out.chunk(len(views)),
            teacher_temp=teacher_temp,
        )
        ibot_loss = self.ibot_criterion(
            teacher_out=teacher_masked_out,
            student_out=student_global_masked_out,
            mask=block_mask,
            teacher_temp=teacher_temp,
        )
        koleo_loss = sum(
            self.koleo_criterion(t) for t in student_global_cls_token.chunk(2)
        )
        
        # Get loss weights from config
        dino_weight = self.training_config["losses"]["dino"]["weight"]
        ibot_weight = self.training_config["losses"]["ibot"]["weight"]
        koleo_weight = self.training_config["losses"]["koleo"]["weight"]
        
        loss = dino_weight * dino_loss + ibot_weight * ibot_loss + koleo_weight * koleo_loss

        # Log losses (both to pytorch lightning and wandb if enabled)
        self.log("train/dino_loss", dino_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/ibot_loss", ibot_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/koleo_loss", koleo_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        lr = self.training_config["optimizer"]["lr"]
        optim = AdamW(self.parameters(), lr=lr)
        return optim

    def on_before_optimizer_step(self, optimizer: AdamW, *args) -> None:
        # Optionally zero out the learning rate of the last layer.
        if self.current_epoch < 1:
            for param_group in optimizer.param_groups:
                if "last_layer" in param_group:
                    param_group["lr"] = 0.0

        # Apply weight decay schedule
        weight_decay_config = self.training_config["weight_decay"]
        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=weight_decay_config["start_value"],
            end_value=weight_decay_config["end_value"],
        )
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                group["weight_decay"] = weight_decay

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Momentum update teacher.
        momentum_config = self.training_config["momentum"]
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=momentum_config["start_value"],
            end_value=momentum_config["end_value"],
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step: extract features and store them for kNN evaluation (if enabled).
        """
        # Only extract features if kNN evaluation is enabled
        if not self.enable_knn_eval:
            return {}
            
        # Handle different batch formats - DINOTransform returns views
        if isinstance(batch[0], list):
            # DINOTransform case: use first global view
            images = batch[0][0]
            labels = batch[1]
            print("Warning: Validation data has DINO transforms, this is probably not what you want.")
        else:
            # Regular case
            images, labels = batch
            
        # Extract features using teacher backbone (frozen, so consistent)
        with torch.no_grad():
            cls_tokens, _ = self.forward_teacher(images)
            
        # Store features and labels for epoch-end kNN evaluation
        self.validation_features.append(cls_tokens.cpu())
        self.validation_labels.append(labels.cpu())
        
        return {"features": cls_tokens, "labels": labels}
    
    def on_validation_epoch_start(self):
        
        if not self.enable_knn_eval or self.trainer.train_dataloader is None:
            return
        
        # temporarily disable the DINO transform
        train_dataset = self.trainer.train_dataloader.dataset
        train_transform = deepcopy(train_dataset.transform) # save for later
        train_dataset.transform = self.trainer.datamodule.norm_only_transform
        
        # Get training dataloader
        train_dl = self.trainer.train_dataloader
        
        # Sample a subset of training data (to avoid memory issues)
        if self.knn_max_train_batches is None:
            max_batches = len(train_dl)
        else:
            max_batches = min(self.knn_max_train_batches, len(train_dl))
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(train_dl), total=max_batches, desc="Collecting train embeds for kNN eval"):
                if i >= max_batches:
                    break
                    
                images, labels = batch # no transform here, so it's the original image
                images = images.to(self.device)
                
                # Extract features using teacher backbone
                cls_tokens, _ = self.forward_teacher(images)
                
                # Store features and labels
                self.train_features.append(cls_tokens.cpu())
                self.train_labels.append(labels.cpu())
        
        # set transformation back to the original one
        train_dataset.transform = train_transform
    
        
    def on_validation_epoch_end(self):
        """
        At the end of validation epoch, perform kNN evaluation using training features (if enabled).
        """
        # Skip kNN evaluation if disabled
        if not self.enable_knn_eval:
            return
            
        if len(self.validation_features) == 0:
            print("No validation features collected, skipping kNN evaluation")
            return
            
        if len(self.train_features) == 0:
            print("No training features available, skipping kNN evaluation")
            return
            
        # Concatenate training and validation features and move to device
        device = self.device
        train_features = torch.cat(self.train_features, dim=0).to(device)
        train_labels = torch.cat(self.train_labels, dim=0).to(device)
        val_features = torch.cat(self.validation_features, dim=0).to(device)
        val_labels = torch.cat(self.validation_labels, dim=0).to(device)
        
        # Perform kNN evaluation: train on training features, test on validation features
        try:
            k = min(self.knn_k, len(train_features))  # Ensure k doesn't exceed train set size
            probs, _ = knn_classifier(train_features, train_labels, val_features, k=k, T=self.knn_temperature, query_batch_size=self.knn_val_chunk_size, train_chunk_size=self.knn_train_chunk_size)
            results = compute_knn_accuracy(probs, val_labels, topk=(1, 5))
            
            # Log results
            self.log("val/knn_top1", results["top1"], prog_bar=True, logger=True, sync_dist=True, on_epoch=True, on_step=False)
            self.log("val/knn_top5", results["top5"], prog_bar=True, logger=True, sync_dist=True, on_epoch=True, on_step=False)
            
            print(f"Epoch {self.current_epoch}: kNN Top1={results['top1']:.2f}%, Top5={results['top5']:.2f}%")
            
        except Exception as e:
            print(f"kNN evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear stored features and labels for next epoch
        self.train_features.clear()
        self.train_labels.clear()
        self.validation_features.clear()
        self.validation_labels.clear()
        
        
        
        
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