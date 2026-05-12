# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bits per byte from cross-entropy loss."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bits_per_byte(
    loss: float,
    base: str = "e",
) -> DescriptiveResult:
    r"""Convert cross-entropy loss to bits per byte (BPB).

    For a model trained with natural log cross-entropy:
    :math:`\\text{BPB} = \\text{loss} / \\ln(2)`.

    :param loss: Cross-entropy loss (natural log base).
    :param base: 'e' if loss is in nats, '2' if already in bits.
    :return: DescriptiveResult with BPB value.
    """
    if loss < 0:
        raise ValueError(f"loss must be >= 0, got {loss}")

    if base == "e":
        bpb = loss / np.log(2)
    elif base == "2":
        bpb = loss
    else:
        raise ValueError(f"base must be 'e' or '2', got {base}")

    return DescriptiveResult(
        name="bits_per_byte",
        value=float(bpb),
        extra={"loss": loss, "base": base},
    )


def cheatsheet() -> str:
    return "bits_per_byte(loss) -> convert CE loss to bits/byte"


bpb = bits_per_byte
