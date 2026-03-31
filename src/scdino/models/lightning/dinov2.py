import torch
from torch import Tensor
from torch.optim import AdamW
from typing import Dict, Any

from tqdm import tqdm
from copy import deepcopy

import lightning as L

from src.scdino.models.backbones.dinov2 import DINOv2 as DINOv2Skeleton
from src.scdino.eval.knn import knn_classifier, compute_knn_accuracy
from src.scdino.models.huggingface import ScDINOConfig, ScDINOModel
from src.scdino.models.lightning.utils import (
    DINOLoss,
    IBOTPatchLoss,
    KoLeoLoss,
    update_momentum,
    cosine_schedule,
    linear_warmup_schedule,
    random_block_mask,
)


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
            ibot_separate_head=self.ibot_separate_head,
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
            center_mode=dino_loss_config["center_mode"],
        )
        self.ibot_criterion = IBOTPatchLoss(
            output_dim=ibot_loss_config["output_dim"],
            teacher_temp=ibot_loss_config["teacher_temp"],
            student_temp=ibot_loss_config["student_temp"],
            center_mode=ibot_loss_config["center_mode"],
            center_momentum=ibot_loss_config["center_momentum"],
        )
        self.koleo_criterion = KoLeoLoss(
            p=koleo_loss_config["p"], eps=koleo_loss_config["eps"]
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
        """For compatibility in the embed.py script"""
        self.teacher_backbone.eval()
        with torch.no_grad():
            embeds = self.teacher_backbone.encode(x)[:, 0]
        self.teacher_backbone.train()
        return embeds

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """Export the teacher backbone as a HuggingFace model."""
        vit_cfg = dict(self.backbone_config.get("vit", {}))

        config = ScDINOConfig(
            model_variant="dinov2",
            backbone_type="vit",
            in_chans=vit_cfg.get("in_chans", 5),
            img_size=vit_cfg.get("img_size", 56),
            patch_size=vit_cfg.get("patch_size", 4),
            embed_dim=vit_cfg.get("embed_dim", 64),
            depth=vit_cfg.get("depth", 12),
            num_heads=vit_cfg.get("num_heads", 8),
            mlp_ratio=vit_cfg.get("mlp_ratio", 4.0),
            reg_tokens=vit_cfg.get("reg_tokens", 0),
        )
        hf_model = ScDINOModel(config)
        hf_model.backbone.load_state_dict(self.teacher_backbone.state_dict())
        hf_model.save_pretrained(save_directory, **kwargs)

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
        assert H * W == sequence_length - n_registered_tokens, (
            f"Unexpected grid size: {H}x{W}, sequence_length {sequence_length}"
        )
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
                teacher_temp_config["warmup_steps"]
                / self.trainer.max_epochs
                * self.trainer.estimated_stepping_batches
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

        loss = (
            dino_weight * dino_loss
            + ibot_weight * ibot_loss
            + koleo_weight * koleo_loss
        )

        # Log losses (both to pytorch lightning and wandb if enabled)
        self.log(
            "train/dino_loss",
            dino_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/ibot_loss",
            ibot_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/koleo_loss",
            koleo_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

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
            print(
                "Warning: Validation data has DINO transforms, this is probably not what you want."
            )
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
        train_transform = deepcopy(train_dataset.transform)  # save for later
        train_dataset.transform = self.trainer.datamodule.norm_only_transform

        # Get training dataloader
        train_dl = self.trainer.train_dataloader

        # Sample a subset of training data (to avoid memory issues)
        if self.knn_max_train_batches is None:
            max_batches = len(train_dl)
        else:
            max_batches = min(self.knn_max_train_batches, len(train_dl))

        with torch.no_grad():
            for i, batch in tqdm(
                enumerate(train_dl),
                total=max_batches,
                desc="Collecting train embeds for kNN eval",
            ):
                if i >= max_batches:
                    break

                images, labels = batch  # no transform here, so it's the original image
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
            k = min(
                self.knn_k, len(train_features)
            )  # Ensure k doesn't exceed train set size
            probs, _ = knn_classifier(
                train_features,
                train_labels,
                val_features,
                k=k,
                T=self.knn_temperature,
                query_batch_size=self.knn_val_chunk_size,
                train_chunk_size=self.knn_train_chunk_size,
            )
            results = compute_knn_accuracy(probs, val_labels, topk=(1, 5))

            # Log results
            self.log(
                "val/knn_top1",
                results["top1"],
                prog_bar=True,
                logger=True,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
            )
            self.log(
                "val/knn_top5",
                results["top5"],
                prog_bar=True,
                logger=True,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
            )

            print(
                f"Epoch {self.current_epoch}: kNN Top1={results['top1']:.2f}%, Top5={results['top5']:.2f}%"
            )

        except Exception as e:
            print(f"kNN evaluation failed: {e}")
            import traceback

            traceback.print_exc()

        # Clear stored features and labels for next epoch
        self.train_features.clear()
        self.train_labels.clear()
        self.validation_features.clear()
        self.validation_labels.clear()
