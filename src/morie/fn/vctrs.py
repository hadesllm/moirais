"""Knowledge itself is power. — Francis Bacon"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def weight_init(
    fan_in: int,
    fan_out: int,
    *,
    method: str = "xavier_uniform",
    gain: float = 1.0,
    seed: int = 42,
) -> DescriptiveResult:
    """Generate neural network weight initialization matrices.

    Supports Xavier (Glorot & Bengio, 2010) and He (He et al., 2015).

    Parameters
    ----------
    fan_in : int
        Number of input units.
    fan_out : int
        Number of output units.
    method : str
        Initialization method:
        'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal',
        'lecun_normal', 'orthogonal'.
    gain : float
        Scaling factor.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        With ``value`` = weight matrix (fan_in, fan_out).
    """
    if fan_in < 1 or fan_out < 1:
        raise ValueError("fan_in and fan_out must be positive")

    rng = np.random.default_rng(seed)

    if method == "xavier_uniform":
        limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
        W = rng.uniform(-limit, limit, (fan_in, fan_out))
    elif method == "xavier_normal":
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        W = rng.normal(0, std, (fan_in, fan_out))
    elif method == "he_uniform":
        limit = gain * np.sqrt(6.0 / fan_in)
        W = rng.uniform(-limit, limit, (fan_in, fan_out))
    elif method == "he_normal":
        std = gain * np.sqrt(2.0 / fan_in)
        W = rng.normal(0, std, (fan_in, fan_out))
    elif method == "lecun_normal":
        std = gain * np.sqrt(1.0 / fan_in)
        W = rng.normal(0, std, (fan_in, fan_out))
    elif method == "orthogonal":
        a = rng.standard_normal((fan_in, fan_out))
        u, _, vt = np.linalg.svd(a, full_matrices=False)
        W = gain * (u if fan_in >= fan_out else vt)
    else:
        raise ValueError(f"Unknown method: {method}")

    var = float(np.var(W))
    expected_var = {
        "xavier_uniform": 2.0 / (fan_in + fan_out),
        "xavier_normal": 2.0 / (fan_in + fan_out),
        "he_uniform": 2.0 / fan_in,
        "he_normal": 2.0 / fan_in,
        "lecun_normal": 1.0 / fan_in,
        "orthogonal": gain**2 / max(fan_in, fan_out),
    }.get(method, var)

    return DescriptiveResult(
        name="weight_init",
        value=W,
        extra={
            "method": method,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "variance": var,
            "expected_variance": expected_var * gain**2,
            "gain": gain,
        },
    )


vctrs = weight_init


def cheatsheet() -> str:
    return "Knowledge itself is power. — Francis Bacon"
