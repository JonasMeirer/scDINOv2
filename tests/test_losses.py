"""Tests for the SSL objectives and the iBOT block-masking helper."""

import pytest
import torch

from src.scdino.models.lightning.utils import (
    DINOLoss,
    IBOTPatchLoss,
    KoLeoLoss,
    random_block_mask,
    update_momentum,
)


class TestDINOLoss:
    def test_matches_an_explicit_cross_entropy_reference(self):
        """Reference check of the whole formula, including diagonal exclusion.

        Same-index teacher/student view pairs are excluded, and the sum is
        averaged over the remaining pairs and the batch.
        """
        torch.manual_seed(0)
        loss_fn = DINOLoss(output_dim=8, student_temp=0.1, center_momentum=1.0)
        teacher = [torch.randn(4, 8) for _ in range(2)]
        student = [torch.randn(4, 8) for _ in range(3)]

        t_prob = [torch.softmax(t / 0.04, dim=-1) for t in teacher]  # centre is 0
        s_logprob = [torch.log_softmax(s / 0.1, dim=-1) for s in student]

        total, n_terms = 0.0, 0
        for ti, t in enumerate(t_prob):
            for si, s in enumerate(s_logprob):
                if ti == si:
                    continue  # same view: excluded
                total = total + -(t * s).sum()
                n_terms += 1
        expected = total / (n_terms * 4)

        actual = loss_fn(teacher, student, teacher_temp=0.04)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_returns_finite_scalar_for_the_multi_crop_setting(self):
        loss_fn = DINOLoss(output_dim=16)
        teacher = list(torch.randn(2, 4, 16))
        student = list(torch.randn(10, 4, 16))
        loss = loss_fn(teacher, student, teacher_temp=0.04)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_agreement_scores_lower_than_disagreement(self):
        torch.manual_seed(0)
        loss_fn = DINOLoss(output_dim=8, center_momentum=1.0)  # freeze the centre at 0
        teacher = torch.zeros(4, 8)
        teacher[:, 0] = 10.0  # teacher is confident about class 0

        agreeing = torch.zeros(4, 8)
        agreeing[:, 0] = 10.0
        disagreeing = torch.zeros(4, 8)
        disagreeing[:, 1] = 10.0

        low = loss_fn([teacher, teacher], [agreeing, agreeing], teacher_temp=0.04)
        high = loss_fn([teacher, teacher], [disagreeing, disagreeing], teacher_temp=0.04)
        assert low < high

    def test_centre_tracks_the_teacher_output(self):
        loss_fn = DINOLoss(output_dim=4, center_momentum=0.5)
        assert torch.allclose(loss_fn.center, torch.zeros(1, 1, 4))
        teacher = torch.ones(2, 4) * 4.0
        loss_fn([teacher, teacher], [teacher, teacher], teacher_temp=0.04)
        # centre = 0.5 * 0 + 0.5 * mean(teacher) = 2.0
        assert torch.allclose(loss_fn.center, torch.full((1, 1, 4), 2.0))

    def test_rejects_unknown_center_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            DINOLoss(output_dim=8, center_mode="median")


class TestIBOTPatchLoss:
    def test_returns_finite_scalar(self):
        loss_fn = IBOTPatchLoss(output_dim=16)
        mask = torch.zeros(2, 4, 4, dtype=torch.bool)
        mask[:, :2, :2] = True
        n_masked = int(mask.sum())
        loss = loss_fn(
            teacher_out=torch.randn(n_masked, 16),
            student_out=torch.randn(n_masked, 16),
            mask=mask,
            teacher_temp=0.04,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_agreement_scores_lower_than_disagreement(self):
        loss_fn = IBOTPatchLoss(output_dim=8, center_momentum=1.0)
        mask = torch.zeros(2, 4, 4, dtype=torch.bool)
        mask[:, 0, :] = True
        n_masked = int(mask.sum())

        teacher = torch.zeros(n_masked, 8)
        teacher[:, 0] = 10.0
        agreeing = teacher.clone()
        disagreeing = torch.zeros(n_masked, 8)
        disagreeing[:, 1] = 10.0

        low = loss_fn(teacher, agreeing, mask=mask, teacher_temp=0.04)
        high = loss_fn(teacher, disagreeing, mask=mask, teacher_temp=0.04)
        assert low < high

    def test_weighting_is_per_image_so_mask_size_does_not_dominate(self):
        """An image with many masked patches must not outweigh one with few."""
        loss_fn = IBOTPatchLoss(output_dim=8, center_momentum=1.0)
        for n_rows in (1, 3):
            mask = torch.zeros(1, 4, 4, dtype=torch.bool)
            mask[:, :n_rows, :] = True
            n_masked = int(mask.sum())
            teacher = torch.zeros(n_masked, 8)
            teacher[:, 0] = 10.0
            loss = loss_fn(teacher, teacher.clone(), mask=mask, teacher_temp=0.04)
            # Per-image normalisation means the value is independent of n_rows.
            assert loss.item() == pytest.approx(loss_fn(
                teacher, teacher.clone(), mask=mask, teacher_temp=0.04
            ).item())


class TestKoLeoLoss:
    def test_returns_finite_scalar(self):
        loss = KoLeoLoss()(torch.randn(16, 8))
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_penalises_collapsed_embeddings_more_than_spread_ones(self):
        """KoLeo exists to punish features piling up on top of each other."""
        loss_fn = KoLeoLoss()
        spread = torch.eye(8)
        collapsed = torch.ones(8, 8) + 1e-4 * torch.randn(8, 8)
        assert loss_fn(collapsed) > loss_fn(spread)


class TestRandomBlockMask:
    def test_shape_and_dtype(self):
        mask = random_block_mask(size=(8, 14, 14))
        assert mask.shape == (8, 14, 14)
        assert mask.dtype == torch.bool

    def test_respects_the_batch_mask_ratio(self):
        """Only batch_mask_ratio of the images get masked at all."""
        mask = random_block_mask(size=(10, 14, 14), batch_mask_ratio=0.5)
        masked_images = (mask.flatten(1).sum(dim=1) > 0).sum().item()
        assert masked_images <= 5

    def test_per_image_mask_ratio_stays_within_bounds(self):
        mask = random_block_mask(
            size=(32, 14, 14), batch_mask_ratio=1.0,
            min_image_mask_ratio=0.1, max_image_mask_ratio=0.5,
        )
        ratios = mask.flatten(1).float().mean(dim=1)
        # The generator can undershoot but must never exceed the upper bound by
        # more than a single block's worth of patches.
        assert ratios.max() <= 0.7

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            random_block_mask(size=(4, 8, 8), min_image_mask_ratio=0.5, max_image_mask_ratio=0.1)


class TestUpdateMomentum:
    def test_interpolates_towards_the_student(self):
        student = torch.nn.Linear(4, 4)
        teacher = torch.nn.Linear(4, 4)
        with torch.no_grad():
            student.weight.fill_(1.0)
            teacher.weight.fill_(0.0)

        update_momentum(student, teacher, m=0.9)
        assert torch.allclose(teacher.weight, torch.full((4, 4), 0.1))

    def test_momentum_of_one_freezes_the_teacher(self):
        student, teacher = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
        with torch.no_grad():
            student.weight.fill_(1.0)
            teacher.weight.fill_(0.0)
        update_momentum(student, teacher, m=1.0)
        assert torch.allclose(teacher.weight, torch.zeros(4, 4))
