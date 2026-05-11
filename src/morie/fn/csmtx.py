# morie.fn — function file (hadesllm/morie)
"""Random sensing matrix generation for compressed sensing."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I know what I have to do. -- Kylo Ren"


def sensing_matrix(
    m: int,
    n: int,
    type: str = "gaussian",
    seed: int = 42,
    **kwargs,
) -> DescriptiveResult:
    """
    Generate a random sensing matrix for compressed sensing.

    Supported types:

    - **gaussian**: i.i.d. :math:`\\mathcal{N}(0, 1/m)` entries
    - **bernoulli**: i.i.d. :math:`\\pm 1/\\sqrt{m}` entries
    - **partial_fourier**: random rows of the DFT matrix (normalized)

    Gaussian and Bernoulli matrices satisfy RIP with high probability
    when :math:`m = O(s \\log(n/s))`.

    :param m: Number of measurements (rows).
    :param n: Signal dimension (columns).
    :param type: Matrix type. Default ``"gaussian"``.
    :param seed: Random seed. Default 42.
    :return: DescriptiveResult with sensing matrix and coherence.
    :raises ValueError: If m < 1 or n < 1 or type unknown.

    References
    ----------
    Candes, E. J. & Tao, T. (2006). Near-optimal signal recovery from
    random projections. *IEEE Trans. Inform. Theory*, 52(12), 5406-5425.
    """
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}.")

    rng = np.random.default_rng(seed)

    if type == "gaussian":
        A = rng.standard_normal((m, n)) / np.sqrt(m)
    elif type == "bernoulli":
        A = rng.choice([-1.0, 1.0], size=(m, n)) / np.sqrt(m)
    elif type == "partial_fourier":
        F = np.fft.fft(np.eye(n)) / np.sqrt(n)
        rows = rng.choice(n, size=m, replace=False)
        A = np.real(F[rows, :])
    else:
        raise ValueError(f"Unknown type '{type}'. Use 'gaussian', 'bernoulli', or 'partial_fourier'.")

    norms = np.linalg.norm(A, axis=0)
    An = A / (norms + 1e-300)
    gram = An.T @ An
    np.fill_diagonal(gram, 0)
    coherence = float(np.max(np.abs(gram)))

    return DescriptiveResult(
        name="sensing_matrix",
        value=coherence,
        extra={
            "matrix": A,
            "shape": (m, n),
            "type": type,
            "coherence": coherence,
            "column_norms_mean": float(np.mean(norms)),
            "compression_ratio": m / n,
        },
    )


csmtx = sensing_matrix


def cheatsheet() -> str:
    return "sensing_matrix({}) -> Random sensing matrix generation for compressed sensing."
