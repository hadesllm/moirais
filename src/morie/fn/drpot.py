# morie.fn — function file (hadesllm/morie)
"""Dropout regularization. 'Big results require big ambitions. — Heraclitus' -- Ahsoka Tano"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dropout(
    x: np.ndarray,
    rate: float = 0.5,
    training: bool = True,
    seed: int = 42,
) -> DescriptiveResult:
    """
    Apply inverted dropout regularization.

    During training, randomly zeros elements with probability *rate*
    and scales surviving elements by 1/(1-rate) to preserve expected
    values at test time.

    :param x: Input array.
    :param rate: Dropout probability in [0, 1). Default 0.5.
    :param training: If False, return input unchanged. Default True.
    :param seed: Random seed for reproducibility. Default 42.
    :return: DescriptiveResult with masked output.
    :raises ValueError: If rate not in [0, 1).

    References
    ----------
    Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., &
    Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural
    networks from overfitting. *JMLR*, 15, 1929-1958.
    """
    if not (0 <= rate < 1):
        raise ValueError(f"rate must be in [0, 1), got {rate}.")

    x_arr = np.asarray(x, dtype=np.float64)

    if not training or rate == 0:
        return DescriptiveResult(
            name="Dropout (inference)",
            value=float(np.mean(x_arr)),
            extra={"output": x_arr.copy(), "mask": np.ones_like(x_arr, dtype=int), "rate": rate},
        )

    rng = np.random.default_rng(seed)
    mask = (rng.random(x_arr.shape) >= rate).astype(np.float64)
    output = (x_arr * mask) / (1.0 - rate)
    frac_dropped = float(1.0 - np.mean(mask))

    return DescriptiveResult(
        name="Dropout",
        value=float(np.mean(output)),
        extra={
            "output": output,
            "mask": mask.astype(int),
            "rate": rate,
            "fraction_dropped": frac_dropped,
        },
    )


short = dropout


def cheatsheet() -> str:
    return "dropout({}) -> Dropout regularization. 'Big results require big ambitions. — Heraclitus' -- Ahsoka Tano"
