import torch
from torch import Tensor
from torch.optim import AdamW
from typing import Dict, Any

from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import silhouette_score

import lightning as L

from src.scdino.models.backbones.dino import DINO as DINOSkeleton
from src.scdino.eval.knn import knn_classifier, compute_knn_accuracy
from src.scdino.models.lightning.utils import (
    DINOLoss,
    update_momentum,
    cosine_schedule,
    linear_warmup_schedule,
)
from src.scdino.models.huggingface import ScDINOConfig, ScDINOModel


class DINO(L.LightningModule):
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
        self.knn_eval_config = training.knn_eval
        self.training_config = training

        model = DINOSkeleton(
            backbone_config=self.backbone_config,
            dino_head_config=self.dino_head_config,
        )

        self.teacher_backbone = model.teacher_backbone
        self.student_backbone = model.student_backbone
        self.teacher_head = model.teacher_head
        self.student_head = model.student_head

        # Loss
        dino_loss_config = self.training_config["losses"]["dino"]

        self.dino_criterion = DINOLoss(
            output_dim=dino_loss_config["output_dim"],
            student_temp=dino_loss_config["student_temp"],
            center_momentum=dino_loss_config["center_momentum"],
            center_mode=dino_loss_config["center_mode"],
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
        """Forward pass through model."""
        return self.teacher_backbone(x)

    def encode(self, x: Tensor) -> Tensor:
        """For compatibility in the embed.py script"""
        self.teacher_backbone.eval()
        with torch.no_grad():
            embeds = self.teacher_backbone(x).flatten(start_dim=1)
        self.teacher_backbone.train()
        return embeds

    @torch.no_grad()
    def get_last_selfattention(self, x: Tensor) -> Tensor:
        """Return last-block attention probs of the teacher backbone.

        Shape: ``(B, num_heads, N, N)``. Only supported for ViT backbones.
        See ``vit_get_last_selfattention``.
        """
        from src.scdino.models.backbones.dino import vit_get_last_selfattention

        self.teacher_backbone.eval()
        return vit_get_last_selfattention(self.teacher_backbone, x)

    @torch.no_grad()
    def get_cls_attention_map(self, x: Tensor, head_fusion: str = "mean") -> Tensor:
        """Return CLS->patch attention heatmap of the teacher backbone.

        See ``vit_get_cls_attention_map``.
        """
        from src.scdino.models.backbones.dino import vit_get_cls_attention_map

        self.teacher_backbone.eval()
        return vit_get_cls_attention_map(
            self.teacher_backbone, x, head_fusion=head_fusion
        )

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """Export the teacher backbone as a HuggingFace model."""
        backbone_type = self.backbone_config["type"]
        vit_cfg = self.backbone_config.get("vit", {})
        resnet_cfg = self.backbone_config.get("resnet", {})
        params = dict(vit_cfg) if backbone_type == "vit" else dict(resnet_cfg)

        config = ScDINOConfig(
            model_variant="dino",
            backbone_type=backbone_type,
            in_chans=params.get("in_chans", 5),
            img_size=params.get("img_size", 56),
            patch_size=params.get("patch_size", 4),
            embed_dim=params.get("embed_dim", 64),
            depth=params.get("depth", 12),
            num_heads=params.get("num_heads", 8),
            mlp_ratio=params.get("mlp_ratio", 4.0),
            reg_tokens=params.get("reg_tokens", 0),
            stem_width=params.get("stem_width", 32),
        )
        hf_model = ScDINOModel(config)
        hf_model.backbone.load_state_dict(self.teacher_backbone.state_dict())
        hf_model.save_pretrained(save_directory, **kwargs)

    def forward_teacher(self, x: Tensor) -> Tensor:
        """Forward pass through teacher model."""
        features = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(features)
        return z

    def forward_student(self, x: Tensor) -> Tensor:
        """Forward pass through student model with head."""
        features = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(features)
        return z

    def training_step(
        self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        views, _ = batch[0], batch[1]

        # Move views to device
        views = [view.to(self.device) for view in views]
        global_views = views[:2]  # First two views are global

        # Teacher forward (with gradient disabled)
        with torch.no_grad():
            teacher_out = [self.forward_teacher(view) for view in global_views]

        # Student forward for all views
        student_out = [self.forward_student(view) for view in views]

        teacher_temp_config = self.training_config["teacher_temp"]
        teacher_temp = linear_warmup_schedule(
            step=self.trainer.global_step,
            warmup_steps=teacher_temp_config["warmup_steps"],
            start_value=teacher_temp_config["start_value"],
            end_value=teacher_temp_config["end_value"],
        )
        loss = self.dino_criterion(
            teacher_out=teacher_out,
            student_out=student_out,
            teacher_temp=teacher_temp,
        )

        # Log loss (both to pytorch lightning and wandb if enabled)
        self.log(
            "train/dino_loss",
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

    def on_after_backward(self):
        """Cancel gradients for the last layer during warmup."""
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher networks with momentum."""
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
            features = self.teacher_backbone(images).flatten(start_dim=1)

        # Store features and labels for epoch-end kNN evaluation
        self.validation_features.append(features.cpu())
        self.validation_labels.append(labels.cpu())

        return {"features": features, "labels": labels}

    @staticmethod
    def _get_transformable_dataset(dataset):
        """Unwrap Subset to get the underlying dataset that holds .transform."""
        from torch.utils.data import Subset

        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        return dataset

    def on_validation_epoch_start(self):
        if self.trainer.sanity_checking:
            return

        if not self.enable_knn_eval:
            return

        train_dl = self.trainer.train_dataloader
        if train_dl is None:
            train_dl = self.trainer.datamodule.train_dataloader(shuffle=True)

        # temporarily disable the DINO transform
        train_dataset = self._get_transformable_dataset(train_dl.dataset)
        train_transform = deepcopy(train_dataset.transform)
        train_dataset.transform = self.trainer.datamodule.norm_only_transform

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

                images, labels = batch
                images = images.to(self.device)

                # Extract features using teacher backbone
                features = self.teacher_backbone(images).flatten(start_dim=1)

                # Store features and labels
                self.train_features.append(features.cpu())
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

            # Log whichever top-k values were computed for this dataset
            # (top-5 is not for datasets with < 5 classes).
            for key, value in results.items():
                self.log(
                    f"val/knn_{key}",
                    value,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    on_epoch=True,
                    on_step=False,
                )

            summary = ", ".join(
                f"{key.capitalize()}={value:.2f}%" for key, value in results.items()
            )
            print(f"Epoch {self.current_epoch}: kNN {summary}")

        except Exception as e:
            print(f"kNN evaluation failed: {e}")
            import traceback

            traceback.print_exc()

        # Silhouette score on validation embeddings
        try:
            val_features_np = val_features.cpu().numpy()
            val_labels_np = val_labels.cpu().numpy()

            n_unique = len(set(val_labels_np.tolist()))
            if n_unique < 2:
                print("Silhouette score requires >= 2 classes, skipping")
            else:
                sil_score = silhouette_score(
                    val_features_np,
                    val_labels_np,
                    sample_size=min(10_000, len(val_features_np)),
                    random_state=42,
                )
                self.log(
                    "val/silhouette",
                    sil_score,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    on_epoch=True,
                    on_step=False,
                )
                print(f"Epoch {self.current_epoch}: Silhouette Score={sil_score:.4f}")
        except Exception as e:
            print(f"Silhouette score computation failed: {e}")
            import traceback

            traceback.print_exc()

        # Clear stored features and labels for next epoch
        self.train_features.clear()
        self.train_labels.clear()
        self.validation_features.clear()
        self.validation_labels.clear()
