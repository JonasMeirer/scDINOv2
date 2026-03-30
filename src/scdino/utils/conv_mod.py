import torch
import torch.nn as nn


def conv_mod(old_conv, out_channels: int):
    # create new conv with 5 input channels
    new_conv = nn.Conv2d(
        in_channels=out_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    # ---- weight adaptation ----
    with torch.no_grad():
        old_weight = old_conv.weight  # shape: (768, 3, 16, 16)

        # average over input channels
        avg_weight = old_weight.mean(dim=1, keepdim=True)  # (768, 1, 16, 16)

        # repeat to 5 channels
        new_weight = avg_weight.repeat(1, out_channels, 1, 1)  # (768, 5, 16, 16)

        new_conv.weight.copy_(new_weight)

        # copy bias if exists
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    # move new layer to same device and dtype as old layer
    new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)

    return new_conv
