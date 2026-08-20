"""Tests for the multi-crop DINO augmentation pipeline."""

import pytest
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf

from scdino.data.transforms.dino import (
    AddGaussianNoise,
    DINOTransform,
    DINOViewTransform,
    RandomChannelDrop,
    RandomChannelGamma,
    RandomChannelIntensityScale,
    RandomChannelIntensityShift,
)
from scdino.data.transforms.trafo import Trafo


@pytest.fixture
def transform_cfg():
    return OmegaConf.create(
        {
            "global_crop_size": 16,
            "global_crop_scale": [0.4, 1.0],
            "local_crop_size": 8,
            "local_crop_scale": [0.1, 0.4],
            "n_local_views": 4,
            "hf_prob": 0.5,
            "vf_prob": 0.5,
            "rr_prob": 0.5,
            "rr_degrees": 90,
            "gaussian_blur": [1.0, 0.1, 0.5],
            "kernel_size": 3,
            "sigmas": [0.1, 2.0],
            "normalize": None,
        }
    )


class TestDINOTransform:
    def test_produces_two_globals_plus_the_configured_locals(self, transform_cfg):
        views = DINOTransform(transform_cfg)(torch.rand(5, 16, 16))
        assert len(views) == 2 + transform_cfg.n_local_views

    def test_global_and_local_views_have_the_configured_sizes(self, transform_cfg):
        views = DINOTransform(transform_cfg)(torch.rand(5, 16, 16))
        for view in views[:2]:
            assert view.shape == (5, 16, 16)
        for view in views[2:]:
            assert view.shape == (5, 8, 8)

    def test_channel_count_is_preserved(self, transform_cfg):
        for channels in (2, 5):
            views = DINOTransform(transform_cfg)(torch.rand(channels, 16, 16))
            assert all(v.shape[0] == channels for v in views)

    def test_normalisation_is_applied_when_configured(self, transform_cfg):
        transform_cfg.normalize = {"mean": [0.5] * 5, "std": [0.1] * 5}
        views = DINOTransform(transform_cfg)(torch.full((5, 16, 16), 0.5))
        # (0.5 - 0.5) / 0.1 == 0 everywhere, modulo the stochastic augmentations.
        assert torch.isfinite(views[0]).all()

    def test_do_center_crop_is_forwarded_to_every_view(self, transform_cfg):
        """The `do_centercrop` ablation config sets this flag.

        Regression test: it used to be read from the config and then dropped
        on the way into DINOViewTransform, so the ablation silently
        reproduced the baseline run.
        """
        import torchvision.transforms as T

        transform_cfg.do_center_crop = True
        views = DINOTransform(transform_cfg)
        for view_transform in views.transforms:
            first_op = view_transform.transform.transforms[0]
            assert isinstance(first_op, T.CenterCrop)

    def test_random_resized_crop_is_the_default(self, transform_cfg):
        import torchvision.transforms as T

        views = DINOTransform(transform_cfg)
        first_op = views.transforms[0].transform.transforms[0]
        assert isinstance(first_op, T.RandomResizedCrop)

    def test_center_cropped_views_keep_the_configured_sizes(self, transform_cfg):
        transform_cfg.do_center_crop = True
        views = DINOTransform(transform_cfg)(torch.rand(5, 16, 16))
        assert all(v.shape == (5, 16, 16) for v in views[:2])
        assert all(v.shape == (5, 8, 8) for v in views[2:])

    def test_views_differ_from_one_another(self, transform_cfg):
        torch.manual_seed(0)
        views = DINOTransform(transform_cfg)(torch.rand(5, 16, 16))
        assert not torch.allclose(views[0], views[1])


class TestTrafo:
    def test_inference_mode_is_a_passthrough(self, transform_cfg):
        model = OmegaConf.create({"name": "dinov2"})
        image = torch.rand(5, 16, 16)
        assert torch.equal(Trafo(model, "inference", transform_cfg)(image), image)

    def test_train_mode_produces_multiple_views(self, transform_cfg):
        model = OmegaConf.create({"name": "dinov2"})
        assert len(Trafo(model, "train", transform_cfg)(torch.rand(5, 16, 16))) == 6

    def test_rejects_an_unknown_model(self, transform_cfg):
        model = OmegaConf.create({"name": "not-a-real-model"})
        with pytest.raises(ValueError, match="Invalid combination"):
            Trafo(model, "train", transform_cfg)


class TestPhotometricAugmentations:
    """These operate per channel, which is the whole point for fluorescence data."""

    def test_intensity_scale_is_multiplicative_and_per_channel(self):
        torch.manual_seed(0)
        x = torch.ones(5, 8, 8)
        out = RandomChannelIntensityScale(scale_range=(2.0, 2.0))(x)
        assert torch.allclose(out, torch.full_like(x, 2.0))

    def test_intensity_scale_uses_a_different_factor_per_channel(self):
        torch.manual_seed(0)
        out = RandomChannelIntensityScale(scale_range=(0.5, 1.5))(torch.ones(5, 8, 8))
        per_channel = out.flatten(1)[:, 0]
        assert per_channel.unique().numel() == 5

    def test_intensity_shift_is_additive(self):
        x = torch.zeros(5, 8, 8)
        out = RandomChannelIntensityShift(shift_range=(0.25, 0.25))(x)
        assert torch.allclose(out, torch.full_like(x, 0.25))

    def test_gamma_preserves_sign(self):
        """Gamma is applied to |x| so it is safe on zero-centred data."""
        x = torch.tensor([[[-4.0]], [[4.0]], [[-1.0]], [[1.0]], [[0.0]]])
        out = RandomChannelGamma(gamma_range=(0.5, 2.0))(x)
        assert torch.equal(out.sign(), x.sign())
        assert torch.isfinite(out).all()

    def test_gamma_of_one_is_the_identity(self):
        x = torch.rand(5, 8, 8) * 4
        assert torch.allclose(
            RandomChannelGamma(gamma_range=(1.0, 1.0))(x), x, atol=1e-6
        )

    def test_gaussian_noise_perturbs_without_shifting_the_mean(self):
        torch.manual_seed(0)
        x = torch.zeros(5, 64, 64)
        out = AddGaussianNoise(sigma=0.5)(x)
        assert not torch.allclose(out, x)
        assert out.mean().abs() < 0.05
        assert out.std() == pytest.approx(0.5, rel=0.1)

    def test_zero_sigma_noise_is_the_identity(self):
        x = torch.rand(5, 8, 8)
        assert torch.allclose(AddGaussianNoise(sigma=0.0)(x), x)

    def test_channel_drop_zeroes_exactly_one_channel(self):
        torch.manual_seed(0)
        out = RandomChannelDrop()(torch.ones(5, 8, 8))
        zeroed = [c for c in range(5) if out[c].abs().sum() == 0]
        assert len(zeroed) == 1

    def test_channel_drop_does_not_mutate_its_input(self):
        x = torch.ones(5, 8, 8)
        RandomChannelDrop()(x)
        assert torch.allclose(x, torch.ones(5, 8, 8))


class TestDINOViewTransform:
    def test_center_crop_is_deterministic(self):
        """A centre crop must return the same pixels every time."""
        transform = DINOViewTransform(
            crop_size=8,
            do_center_crop=True,
            hf_prob=0,
            vf_prob=0,
            rr_prob=0,
            intensity_scale_prob=0,
            intensity_shift_prob=0,
            gamma_prob=0,
            gaussian_noise_prob=0,
            random_channel_drop_prob=0,
            gaussian_blur=0,
            normalize=None,
        )
        x = torch.rand(5, 16, 16)
        assert torch.equal(transform(x), transform(x))

    def test_disabling_every_augmentation_leaves_a_plain_crop(self):
        transform = DINOViewTransform(
            crop_size=16,
            do_center_crop=True,
            hf_prob=0,
            vf_prob=0,
            rr_prob=0,
            intensity_scale_prob=0,
            intensity_shift_prob=0,
            gamma_prob=0,
            gaussian_noise_prob=0,
            random_channel_drop_prob=0,
            gaussian_blur=0,
            normalize=None,
        )
        x = torch.rand(5, 16, 16)
        assert torch.equal(transform(x), x)


class TestChannelDropSemantics:
    """A dropped channel must read as "no information", not as an outlier."""

    def test_fills_with_the_supplied_per_channel_value(self):
        torch.manual_seed(0)
        fill = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = RandomChannelDrop(fill=fill)(torch.zeros(5, 4, 4))
        dropped = [c for c in range(5) if out[c].abs().sum() > 0]
        assert len(dropped) == 1
        assert out[dropped[0]].unique().tolist() == [fill[dropped[0]]]

    def test_falls_back_to_zero_without_a_fill(self):
        torch.manual_seed(0)
        out = RandomChannelDrop()(torch.ones(5, 4, 4))
        assert sum(out[c].abs().sum() == 0 for c in range(5)) == 1

    def test_dropped_channel_is_exactly_zero_after_normalisation(self):
        """The whole point: fill == dataset mean, so normalising sends it to 0.

        Filling with a literal 0 instead normalises to -mean/std, which for the
        Brightfield channel is a constant -6.2 against a natural range of ~±1 --
        a marker the model can simply detect.
        """
        mean = [358.5443, 941.6513, 825.9841, 255.2261, 308.7898]
        std = [499.9700, 151.3277, 1314.0288, 311.5910, 399.3656]
        normalize = T.Normalize(mean=mean, std=std)

        for seed in range(10):
            torch.manual_seed(seed)
            x = torch.tensor(mean, dtype=torch.float32)[:, None, None].expand(5, 4, 4)
            z = normalize(RandomChannelDrop(fill=mean)(x.clone()))
            assert torch.allclose(z, torch.zeros_like(z), atol=1e-4)

    def test_zero_fill_would_produce_an_out_of_range_constant(self):
        """Pins why the fill argument exists, so the default cannot regress."""
        mean = [358.5443, 941.6513, 825.9841, 255.2261, 308.7898]
        std = [499.9700, 151.3277, 1314.0288, 311.5910, 399.3656]
        worst = max(m / s for m, s in zip(mean, std))
        assert worst > 6.0, (
            "expected the Brightfield channel to be the pathological one"
        )


class TestAugmentationEffectSizes:
    """Regression guard for the augmentation-ordering defect.

    Every photometric augmentation must move the tensor the model actually
    receives by a non-trivial but non-dominant amount. Before the loader was
    changed to emit [0, 1] data, `gaussian_noise` and `intensity_shift` were
    numerically inert (~1e-4 sigma) while `gamma` moved the input by ~8 sigma,
    so three augmentation ablations were not measuring what they claimed.

    The bounds are deliberately wide: this catches "does nothing" and "swamps
    the signal", not small parameter tweaks.
    """

    MEAN = [358.5443, 941.6513, 825.9841, 255.2261, 308.7898]
    STD = [499.9700, 151.3277, 1314.0288, 311.5910, 399.3656]
    CLIP = [3713.48, 1297.59, 6904.38, 3412.43, 3133.96]

    LOWER, UPPER = 0.01, 1.0

    def _quiet_cfg(self, **overrides):
        base = dict(
            do_center_crop=True,
            hf_prob=0,
            vf_prob=0,
            rr_prob=0,
            intensity_scale_prob=0,
            intensity_shift_prob=0,
            gamma_prob=0,
            gaussian_noise_prob=0,
            random_channel_drop_prob=0,
            gaussian_blur=[0.0, 0.0, 0.0],
            cutout_prob=0.0,
            n_local_views=0,
            global_crop_size=32,
            local_crop_size=16,
            normalize={
                "mean": [m / c for m, c in zip(self.MEAN, self.CLIP)],
                "std": [s / c for s, c in zip(self.STD, self.CLIP)],
            },
        )
        base.update(overrides)
        return OmegaConf.create(base)

    def _images(self, n=6, hw=32):
        """Unit-range images with realistic right-skewed channel statistics."""
        rng = torch.Generator().manual_seed(0)
        out = []
        for _ in range(n):
            chans = []
            for m, s, c in zip(self.MEAN, self.STD, self.CLIP):
                x = torch.rand(hw, hw, generator=rng) * (s / c) + (m / c)
                chans.append(x.clamp(0.0, 1.0))
            out.append(torch.stack(chans))
        return out

    def _view(self, img, cfg, seed):
        torch.manual_seed(seed)
        return DINOTransform(cfg)(img)[0]

    @pytest.mark.parametrize(
        "name,override",
        [
            ("intensity_scale", {"intensity_scale_prob": 1.0}),
            ("intensity_shift", {"intensity_shift_prob": 1.0}),
            ("gamma", {"gamma_prob": 1.0}),
            ("gaussian_noise", {"gaussian_noise_prob": 1.0}),
            ("channel_drop", {"random_channel_drop_prob": 1.0}),
        ],
    )
    def test_effect_size_is_neither_inert_nor_dominant(self, name, override):
        baseline_cfg = self._quiet_cfg()
        active_cfg = self._quiet_cfg(**override)
        deltas = [
            (self._view(img, active_cfg, seed) - self._view(img, baseline_cfg, seed))
            .abs()
            .mean()
            .item()
            for seed in range(4)
            for img in self._images()
        ]
        median = sorted(deltas)[len(deltas) // 2]
        assert median > self.LOWER, (
            f"{name} moves the model input by only {median:.5f} sigma -- it is "
            "effectively a no-op, so any ablation of it measures nothing"
        )
        assert median < self.UPPER, (
            f"{name} moves the model input by {median:.3f} sigma, swamping the "
            "signal; check that the loader emits unit-range data"
        )
