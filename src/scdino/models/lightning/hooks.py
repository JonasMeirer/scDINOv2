"""Shared Lightning training behaviour for the scDINO modules.

This is original scDINO code (MIT). It deliberately lives outside
``models/lightning/utils.py``, which is vendored from Lightly, so the vendored
file can be re-synced from upstream without merge pain.
"""

from torch import nn


def freeze_last_layer_gradients(
    head: nn.Module, current_epoch: int, freeze_epochs: int = 1
) -> None:
    """Drop gradients for a projection head's ``last_layer`` during warmup.

    DINO freezes the output layer of the projection head for the first epoch:
    the head is randomly initialised, so its early gradients are large and
    noisy, and letting them through destabilises the student/teacher dynamic.

    This must run *after* backward and *before* the optimizer step, so call it
    from ``LightningModule.on_after_backward``. Setting ``.grad`` to ``None``
    (rather than zeroing it) also stops optimizers with weight decay or
    momentum from updating the parameter on the side.

    Args:
        head:
            Either a projection head exposing ``last_layer``, or a container
            such as ``DINOv2Head`` exposing ``dino_head`` / ``ibot_head``. When
            ``ibot_separate_head`` is False those two attributes are the same
            module, which is handled.
        current_epoch:
            The epoch about to be stepped.
        freeze_epochs:
            Number of leading epochs to freeze for. ``0`` disables freezing.
    """
    if current_epoch >= freeze_epochs:
        return

    seen: set[int] = set()
    for sub_head in _projection_heads(head):
        if id(sub_head) in seen:
            continue  # dino_head and ibot_head are the same module when tied
        seen.add(id(sub_head))
        for param in sub_head.last_layer.parameters():
            param.grad = None


def _projection_heads(head: nn.Module):
    """Yield the projection heads carrying a ``last_layer``."""
    found = False
    for name in ("dino_head", "ibot_head"):
        sub_head = getattr(head, name, None)
        if sub_head is not None and hasattr(sub_head, "last_layer"):
            found = True
            yield sub_head
    if not found:
        if not hasattr(head, "last_layer"):
            raise AttributeError(
                f"{type(head).__name__} exposes neither a `last_layer` nor "
                "`dino_head`/`ibot_head` sub-heads, so the DINO warmup freeze "
                "cannot be applied."
            )
        yield head
