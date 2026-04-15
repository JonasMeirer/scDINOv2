import torch
import torch.nn as nn

FLAVORS = ("mean", "pick_1", "pick_2", "pick_3")

_PICK_INDEX = {"pick_1": 0, "pick_2": 1, "pick_3": 2}


def conv_mod(old_conv, out_channels: int, flavor: str = "mean"):
    if flavor not in FLAVORS:
        raise ValueError(f"Unknown flavor {flavor!r}, expected one of {FLAVORS}")

    new_conv = nn.Conv2d(
        in_channels=out_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    with torch.no_grad():
        old_weight = old_conv.weight  # (out, 3, kH, kW)

        if flavor == "mean":
            seed_weight = old_weight.mean(dim=1, keepdim=True)
        else:
            idx = _PICK_INDEX[flavor]
            seed_weight = old_weight[:, idx : idx + 1, :, :]

        new_weight = seed_weight.repeat(1, out_channels, 1, 1)
        new_conv.weight.copy_(new_weight)

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)

    return new_conv
