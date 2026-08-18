"""Tests for adapting 3-channel pretrained backbones to N-channel microscopy input."""

import pytest
import torch
import torch.nn as nn

from scdino.utils.conv_mod import FLAVORS, conv_mod
from scdino.utils.per_channel_wrapper import PerChannelWrapper


@pytest.fixture
def rgb_conv():
    torch.manual_seed(0)
    return nn.Conv2d(3, 8, kernel_size=4, stride=4)


class TestConvMod:
    @pytest.mark.parametrize("flavor", FLAVORS)
    def test_accepts_n_channel_input_and_keeps_the_output_shape(self, rgb_conv, flavor):
        adapted = conv_mod(rgb_conv, out_channels=5, flavor=flavor)
        out = adapted(torch.randn(2, 5, 16, 16))
        assert out.shape == (2, 8, 4, 4)

    @pytest.mark.parametrize("flavor", FLAVORS)
    def test_preserves_kernel_geometry(self, rgb_conv, flavor):
        adapted = conv_mod(rgb_conv, out_channels=5, flavor=flavor)
        assert adapted.in_channels == 5
        assert adapted.out_channels == rgb_conv.out_channels
        assert adapted.kernel_size == rgb_conv.kernel_size
        assert adapted.stride == rgb_conv.stride
        assert adapted.padding == rgb_conv.padding

    def test_preserves_the_bias(self, rgb_conv):
        adapted = conv_mod(rgb_conv, out_channels=5, flavor="mean")
        torch.testing.assert_close(adapted.bias, rgb_conv.bias)

    def test_bias_free_conv_stays_bias_free(self):
        conv = nn.Conv2d(3, 8, 4, 4, bias=False)
        assert conv_mod(conv, out_channels=5, flavor="mean").bias is None

    def test_mean_flavor_seeds_every_channel_with_the_rgb_mean(self, rgb_conv):
        adapted = conv_mod(rgb_conv, out_channels=5, flavor="mean")
        expected = rgb_conv.weight.mean(dim=1, keepdim=True)
        for channel in range(5):
            torch.testing.assert_close(
                adapted.weight[:, channel : channel + 1], expected
            )

    @pytest.mark.parametrize(
        "flavor,index", [("pick_1", 0), ("pick_2", 1), ("pick_3", 2)]
    )
    def test_pick_flavors_seed_from_the_named_rgb_channel(
        self, rgb_conv, flavor, index
    ):
        adapted = conv_mod(rgb_conv, out_channels=5, flavor=flavor)
        expected = rgb_conv.weight[:, index : index + 1]
        for channel in range(5):
            torch.testing.assert_close(
                adapted.weight[:, channel : channel + 1], expected
            )

    def test_rejects_an_unknown_flavor(self, rgb_conv):
        with pytest.raises(ValueError, match="Unknown flavor"):
            conv_mod(rgb_conv, out_channels=5, flavor="bogus")

    def test_identity_case_reproduces_the_original_conv(self, rgb_conv):
        """Adapting 3 channels with pick_1 replicates channel 0 three times, so
        a constant-across-channels input gives the same answer as the original
        conv fed the same constant."""
        adapted = conv_mod(rgb_conv, out_channels=3, flavor="pick_1")
        x = torch.randn(2, 1, 16, 16).expand(-1, 3, -1, -1).contiguous()
        original_w0 = rgb_conv.weight[:, 0:1]
        expected = nn.functional.conv2d(
            x,
            original_w0.expand(-1, 3, -1, -1).contiguous(),
            rgb_conv.bias,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
        )
        torch.testing.assert_close(adapted(x), expected)


class TestPerChannelWrapper:
    @pytest.fixture
    def base_model(self):
        """Stands in for a pretrained 3-channel backbone with a 7-d embedding."""
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 7))

    def test_concat_flavor_stacks_one_embedding_per_channel(self, base_model):
        wrapper = PerChannelWrapper(base_model, lambda o: o, num_channels=5)
        out = wrapper(torch.randn(2, 5, 16, 16))
        assert out.shape == (2, 5 * 7)

    def test_mean_flavor_averages_over_channels(self, base_model):
        wrapper = PerChannelWrapper(
            base_model, lambda o: o, num_channels=5, flavor="mean"
        )
        out = wrapper(torch.randn(2, 5, 16, 16))
        assert out.shape == (2, 7)

    def test_mean_equals_the_average_of_the_concatenated_parts(self, base_model):
        torch.manual_seed(0)
        images = torch.randn(2, 5, 16, 16)
        concat = PerChannelWrapper(base_model, lambda o: o, 5)(images)
        mean = PerChannelWrapper(base_model, lambda o: o, 5, flavor="mean")(images)
        torch.testing.assert_close(mean, concat.view(2, 5, 7).mean(dim=1))

    def test_each_channel_is_expanded_to_pseudo_rgb(self, base_model):
        """The base model only accepts 3 channels; the wrapper must repeat."""
        seen = []

        class Recorder(nn.Module):
            def forward(self, x):
                seen.append(x.shape)
                return base_model(x)

        PerChannelWrapper(Recorder(), lambda o: o, num_channels=5)(
            torch.randn(2, 5, 16, 16)
        )
        assert len(seen) == 5
        assert all(shape[1] == 3 for shape in seen)

    def test_rejects_an_unknown_flavor_at_construction(self, base_model):
        """Fail fast, and with a ValueError rather than a NameError."""
        with pytest.raises(ValueError, match="Unknown flavor"):
            PerChannelWrapper(base_model, lambda o: o, num_channels=5, flavor="bogus")

    def test_error_lists_the_valid_flavors(self, base_model):
        with pytest.raises(ValueError, match="concat"):
            PerChannelWrapper(base_model, lambda o: o, num_channels=5, flavor="bogus")

    def test_runs_without_grad(self, base_model):
        out = PerChannelWrapper(base_model, lambda o: o, 5)(torch.randn(2, 5, 16, 16))
        assert not out.requires_grad
