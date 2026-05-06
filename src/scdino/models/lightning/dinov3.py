"""Lightning module for the DINOv3 backbone.

Mirrors :mod:`src.scdino.models.lightning.dinov2_StrucPerc` (which itself
mirrors :mod:`src.scdino.models.lightning.dinov2`). Only the backbone
import and ``save_pretrained`` are different; the rest of the
training/eval logic is reused as-is, since
:class:`MaskedDinoVisionTransformer` exposes the same public surface as
:class:`MaskedVisionTransformerTIMM`.
"""

import torch
from torch import Tensor
from torch.optim import AdamW
from typing import Dict, Any

from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import silhouette_score

import lightning as L

from src.scdino.models.backbones.dinov3 import DINOv3 as DINOv3Skeleton
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


class DINOv3(L.LightningModule):
    def __init__(
        self,
        name: str,
        architecture: Dict[str, Any],
        training: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.name = name
        self.backbone_config = architecture.backbone
        self.dino_head_config = architecture.dino_head
        self.ibot_head_config = architecture.ibot_head
        self.ibot_separate_head = architecture.ibot_separate_head
        self.knn_eval_config = training.knn_eval
        self.training_config = training

        model = DINOv3Skeleton(
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
            student_temp=dino_loss_config["student_temp"],
            center_momentum=dino_loss_config["center_momentum"],
            center_mode=dino_loss_config["center_mode"],
        )
        self.ibot_criterion = IBOTPatchLoss(
            output_dim=ibot_loss_config["output_dim"],
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

    @torch.no_grad()
    def get_last_selfattention(self, x: Tensor) -> Tensor:
        self.teacher_backbone.eval()
        return self.teacher_backbone.get_last_selfattention(x)

    @torch.no_grad()
    def get_cls_attention_map(self, x: Tensor, head_fusion: str = "mean") -> Tensor:
        self.teacher_backbone.eval()
        return self.teacher_backbone.get_cls_attention_map(x, head_fusion=head_fusion)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """Export the teacher backbone as a HuggingFace ScDINOModel."""
        cfg = dict(self.backbone_config.get("dinov3", {}))

        config = ScDINOConfig(
            model_variant="dinov3",
            backbone_type="dinov3",
            in_chans=cfg.get("in_chans", 5),
            img_size=cfg.get("img_size", 50),
            patch_size=cfg.get("patch_size", 5),
            embed_dim=cfg.get("embed_dim", 64),
            depth=cfg.get("depth", 12),
            num_heads=cfg.get("num_heads", 8),
            mlp_ratio=cfg.get("ffn_ratio", cfg.get("mlp_ratio", 4.0)),
            reg_tokens=cfg.get("n_storage_tokens", cfg.get("reg_tokens", 0)),
            qkv_bias=cfg.get("qkv_bias", True),
            init_values=cfg.get("layerscale_init", cfg.get("init_values", 1e-5)),
            # 2D RoPE
            rope_base=cfg.get("rope_base", 100.0),
            rope_min_period=cfg.get("rope_min_period", None),
            rope_max_period=cfg.get("rope_max_period", None),
            rope_normalize_coords=cfg.get("rope_normalize_coords", "separate"),
            rope_shift_coords=cfg.get("rope_shift_coords", None),
            rope_jitter_coords=cfg.get("rope_jitter_coords", None),
            rope_rescale_coords=cfg.get("rope_rescale_coords", None),
            # DINOv3-specific
            ffn_layer=cfg.get("ffn_layer", "mlp"),
            norm_layer=cfg.get("norm_layer", "layernorm"),
            rope_dtype=cfg.get("rope_dtype", "fp32"),
            proj_bias=cfg.get("proj_bias", True),
            ffn_bias=cfg.get("ffn_bias", True),
            mask_k_bias=cfg.get("mask_k_bias", False),
            untie_cls_and_patch_norms=cfg.get("untie_cls_and_patch_norms", False),
            untie_global_and_local_cls_norm=cfg.get(
                "untie_global_and_local_cls_norm", False
            ),
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

        # iBOT mask on the patch grid (H * W = num_patches).
        B = len(global_views)
        sequence_length = self.teacher_backbone.sequence_length
        mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)
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
            warmup_steps=teacher_temp_config["warmup_steps"],
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

        dino_weight = self.training_config["losses"]["dino"]["weight"]
        ibot_weight = self.training_config["losses"]["ibot"]["weight"]
        koleo_weight = self.training_config["losses"]["koleo"]["weight"]

        loss = (
            dino_weight * dino_loss
            + ibot_weight * ibot_loss
            + koleo_weight * koleo_loss
        )

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
        if self.current_epoch < 1:
            for param_group in optimizer.param_groups:
                if "last_layer" in param_group:
                    param_group["lr"] = 0.0

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
        if not self.enable_knn_eval:
            return {}

        if isinstance(batch[0], list):
            images = batch[0][0]
            labels = batch[1]
            print(
                "Warning: Validation data has DINO transforms, this is probably not what you want."
            )
        else:
            images, labels = batch

        with torch.no_grad():
            cls_tokens, _ = self.forward_teacher(images)

        self.validation_features.append(cls_tokens.cpu())
        self.validation_labels.append(labels.cpu())

        return {"features": cls_tokens, "labels": labels}

    @staticmethod
    def _get_transformable_dataset(dataset):
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

        train_dataset = self._get_transformable_dataset(train_dl.dataset)
        train_transform = deepcopy(train_dataset.transform)
        train_dataset.transform = self.trainer.datamodule.norm_only_transform

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

                cls_tokens, _ = self.forward_teacher(images)

                self.train_features.append(cls_tokens.cpu())
                self.train_labels.append(labels.cpu())

        train_dataset.transform = train_transform

    def on_validation_epoch_end(self):
        if not self.enable_knn_eval:
            return

        if len(self.validation_features) == 0:
            print("No validation features collected, skipping kNN evaluation")
            return

        if len(self.train_features) == 0:
            print("No training features available, skipping kNN evaluation")
            return

        device = self.device
        train_features = torch.cat(self.train_features, dim=0).to(device)
        train_labels = torch.cat(self.train_labels, dim=0).to(device)
        val_features = torch.cat(self.validation_features, dim=0).to(device)
        val_labels = torch.cat(self.validation_labels, dim=0).to(device)

        try:
            k = min(self.knn_k, len(train_features))
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
                print(
                    f"Epoch {self.current_epoch}: Silhouette Score={sil_score:.4f}"
                )
        except Exception as e:
            print(f"Silhouette score computation failed: {e}")
            import traceback

            traceback.print_exc()

        self.train_features.clear()
        self.train_labels.clear()
        self.validation_features.clear()
        self.validation_labels.clear()
