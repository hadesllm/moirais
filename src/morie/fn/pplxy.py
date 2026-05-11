# morie.fn — function file (hadesllm/morie)
"""Perplexity from cross-entropy loss."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def perplexity(
    loss: float,
) -> DescriptiveResult:
    """Compute perplexity from cross-entropy loss.

    :math:`\\text{PPL} = \\exp(\\text{loss})`

    :param loss: Cross-entropy loss (natural log base).
    :return: DescriptiveResult with perplexity value.
    """
    if loss < 0:
        raise ValueError(f"loss must be >= 0, got {loss}")

    ppl = float(np.exp(loss))

    return DescriptiveResult(
        name="perplexity",
        value=ppl,
        extra={"loss": loss},
    )


def cheatsheet() -> str:
    return "perplexity(loss) -> exp(loss) perplexity metric"


pplxy = perplexity
