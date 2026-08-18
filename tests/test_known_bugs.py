"""Executable record of known, unfixed defects.

Every test here asserts the behaviour we *want* and is marked
``xfail(strict=True)``. That means:

* while the bug exists, the test xfails and CI stays green;
* the moment someone fixes the bug, the test XPASSes and CI turns **red**,
  telling them to delete the marker and promote the test to a real one.

So this file is a to-do list that cannot silently rot. Do not add ``xfail``
markers here for anything other than a defect that is genuinely still open.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.scdino.data.transforms.dino import DINOTransform
from src.scdino.eval.knn import compute_knn_accuracy, knn_classifier
from src.scdino.models.lightning.utils import DINOLoss
from src.scdino.utils.per_channel_wrapper import PerChannelWrapper


@pytest.mark.xfail(strict=True, reason="review #5: uses ndarray.T, which reverses every axis")
def test_hwc_to_chw_conversion_must_not_transpose_the_image():
    """`load_tiff` converts (H, W, C) to (C, H, W) with `img.T`.

    `.T` reverses *all* axes, giving (C, W, H): the spatial dimensions are
    swapped. It is invisible on square crops and wrong on everything else.
    The correct operation is `np.transpose(img, (2, 0, 1))`.
    """
    hwc = np.zeros((4, 6, 5))  # H=4, W=6, C=5
    assert hwc.T.shape == (5, 4, 6)


@pytest.mark.xfail(strict=True, reason="review #6: do_center_crop is not forwarded by DINOTransform")
def test_do_center_crop_reaches_the_view_transform():
    """`configs/experiment/augmentation/do_centercrop.yaml` sets this flag, but
    `DINOTransform.__init__` never passes it into `DINOViewTransform`, so the
    ablation silently reproduces the baseline run."""
    import torchvision.transforms as T

    cfg = OmegaConf.create(
        {"do_center_crop": True, "n_local_views": 1, "global_crop_size": 8,
         "local_crop_size": 4, "normalize": None}
    )
    first_op = DINOTransform(cfg).transforms[0].transform.transforms[0]
    assert isinstance(first_op, T.CenterCrop)


@pytest.mark.xfail(strict=True, reason="review #7: 'last_layer' is never a key in an optimizer param group")
def test_last_layer_freeze_hook_can_actually_match_something():
    """`on_before_optimizer_step` looks for `"last_layer" in param_group`.

    Optimizer param groups only ever contain hyperparameter keys (lr, betas,
    eps, weight_decay, params, ...), so the branch is dead and the DINO
    warmup requirement to freeze the projection head is not enforced for
    DINOv2/DINOv3. Fixing it needs a named param group or an
    `on_after_backward` hook like the one `DINO` already has.
    """
    optimizer = torch.optim.AdamW(nn.Linear(4, 4).parameters(), lr=1e-3)
    assert any("last_layer" in group for group in optimizer.param_groups)


@pytest.mark.xfail(strict=True, reason="review #8: FLAVORS is not imported in per_channel_wrapper")
def test_per_channel_wrapper_rejects_unknown_flavor_with_a_useful_error():
    """The error branch references an undefined `FLAVORS`, so an unknown flavor
    raises `NameError` instead of the intended `ValueError`."""
    model = PerChannelWrapper(nn.Identity(), lambda o: o.flatten(1), 5, flavor="bogus")
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 8, 8))


@pytest.mark.xfail(strict=True, reason="review #9: topk(k) on a final chunk smaller than k")
def test_knn_handles_a_final_train_chunk_smaller_than_k():
    """`knn_classifier` calls `sims.topk(k)` per chunk without clamping k to the
    chunk size, so it raises whenever
    `len(train_features) % train_chunk_size < k`.
    """
    train_x = torch.randn(105, 8)
    train_y = torch.randint(0, 3, (105,))
    probs, _ = knn_classifier(
        train_x, train_y, torch.randn(4, 8), k=20, T=0.07, train_chunk_size=100,
    )
    assert probs.shape == (4, 3)


@pytest.mark.xfail(strict=True, reason="review #10: filter is k <= num_classes, should be k < num_classes")
def test_topk_equal_to_class_count_is_not_reported():
    """top-5 accuracy on a 5-class problem is trivially 100% and must be
    dropped, exactly as top-2 is dropped for a 2-class problem."""
    result = compute_knn_accuracy(torch.rand(50, 5), torch.randint(0, 5, (50,)), topk=(1, 5))
    assert "top5" not in result


@pytest.mark.xfail(strict=True, reason="review #28: configs/train.yaml ships max_train_batches: 1000")
def test_training_is_not_truncated_by_default():
    """The default training config caps every epoch at 1000 batches with only a
    trailing comment to say so, which silently shortens the documented
    `python -m src.scdino.train.run` run."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = (Path(__file__).resolve().parents[1] / "configs").as_posix()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=[])
    assert cfg.max_train_batches is None


@pytest.mark.xfail(strict=True, reason="single-view input divides by zero n_terms")
def test_dino_loss_does_not_return_nan_for_a_degenerate_single_view_batch():
    """With one teacher and one student view every pair is on the excluded
    diagonal, so `n_terms` is 0 and the loss is a silent NaN. Training always
    supplies two global views, so this is latent rather than active, but it
    should raise a clear error instead of poisoning the loss.
    """
    out = torch.randn(4, 8)
    loss = DINOLoss(output_dim=8)([out], [out], teacher_temp=0.04)
    assert torch.isfinite(loss)
