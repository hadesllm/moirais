# morie.fn — function file (hadesllm/morie)
"""He/Kaiming weight initialization for ReLU networks."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["he_initialization"]


def he_initialization(fan_in, fan_out=None, seed: int = 42, mode: str = "normal",
                      deterministic_seed: "int | None" = None):
    r"""He/Kaiming weight initialization.

    For ReLU activations the recommended scheme is:

    .. math::

        W \\sim \\mathcal{N}\\!\\left(0, \\tfrac{2}{n_{in}}\\right)

    (normal) or :math:`W \\sim U[-\\sqrt{6/n_{in}}, \\sqrt{6/n_{in}}]`
    (uniform).

    Parameters
    ----------
    fan_in : int
        Number of input units of the layer.
    fan_out : int, optional
        Number of output units. If ``None``, returns a vector of length
        ``fan_in``.
    seed : int
        RNG seed for reproducibility.
    mode : str
        ``'normal'`` (default) or ``'uniform'``.
    deterministic_seed : int or None, optional
        If given, the SHA-keyed RNG from
        :func:`morie._det_rng.from_seed` is used so Py<->R streams agree
        for the same ``(name, seed)`` pair. Overrides ``seed`` when set.

    Returns
    -------
    result : RichResult
        Keys: ``W`` / ``estimate`` (the weight tensor), ``mean``, ``std``,
        ``shape``.

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into
    rectifiers: surpassing human-level performance on ImageNet
    classification. *ICCV*.
    """
    fan_in = int(fan_in)
    if fan_in <= 0:
        raise ValueError(f"fan_in must be > 0, got {fan_in}.")
    shape = (fan_in,) if fan_out is None else (int(fan_out), fan_in)
    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("heinz", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    if mode == "normal":
        std = np.sqrt(2.0 / fan_in)
        W = rng.normal(0.0, std, size=shape)
    elif mode == "uniform":
        limit = np.sqrt(6.0 / fan_in)
        W = rng.uniform(-limit, limit, size=shape)
    else:
        raise ValueError(f"mode must be 'normal' or 'uniform', got {mode!r}.")

    return RichResult(
        title="He/Kaiming initialization",
        summary_lines=[
            ("fan_in", fan_in),
            ("shape", shape),
            ("mean", float(np.mean(W))),
            ("std", float(np.std(W))),
        ],
        payload={
            "W": W,
            "estimate": W,
            "mean": float(np.mean(W)),
            "std": float(np.std(W)),
            "shape": shape,
            "method": f"He initialization ({mode})",
        },
    )


# CANONICAL TEST
# he_initialization(100, 100, seed=0).std ~ sqrt(2/100) = 0.1414


def cheatsheet():
    return "heinz: He/Kaiming init W ~ N(0, 2/n_in) for ReLU layers"
