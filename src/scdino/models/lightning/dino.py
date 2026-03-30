import torch
from torch import Tensor
from torch.optim import AdamW
from typing import Dict, Any

from tqdm import tqdm
from copy import deepcopy

import lightning as L

from src.scdino.models.backbones.dino import DINO as DINOSkeleton
from src.scdino.eval.knn import knn_classifier, compute_knn_accuracy
from src.scdino.models.lightning.utils import DINOLoss, update_momentum, cosine_schedule


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
        self.ibot_head_config = architecture.ibot_head
        self.ibot_separate_head = architecture.ibot_separate_head
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
            warmup_teacher_temp=dino_loss_config["warmup_teacher_temp"],
            teacher_temp=dino_loss_config["teacher_temp"],
            warmup_teacher_temp_epochs=dino_loss_config["warmup_teacher_temp_epochs"],
            student_temp=dino_loss_config["student_temp"],
            center_momentum=dino_loss_config["center_momentum"],
            center_mode=dino_loss_config["center_mode"]
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
        """ For compatibility in the embed.py script """
        self.teacher_backbone.eval()
        with torch.no_grad():
            embeds = self.teacher_backbone(x).flatten(start_dim=1)
        self.teacher_backbone.train()
        return embeds

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
        
        # Compute DINO loss
        loss = self.dino_criterion(
            teacher_out=teacher_out,
            student_out=student_out,
            epoch=self.current_epoch
        )

        # Log loss (both to pytorch lightning and wandb if enabled)
        self.log("train/dino_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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
            print("Warning: Validation data has DINO transforms, this is probably not what you want.")
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
