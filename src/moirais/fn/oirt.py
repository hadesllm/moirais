# moirais.fn — function file (hadesllm/moirais)
"""Ordinal IRT / mixed factor analysis."""

from __future__ import annotations

from ._containers import DescriptiveResult


def ordinal_irt_model(
    Y,
    n_dims: int = 1,
    n_samples: int = 500,
    burn_in: int = 100,
    seed: int = 42,
) -> DescriptiveResult:
    """Ordinal IRT with probit link (Quinn 2004).

    :param Y: (n x m) ordinal response matrix.
    :param n_dims: Number of latent dimensions.
    :param n_samples: MCMC samples.
    :param burn_in: Burn-in samples.
    :param seed: Random seed.
    :return: DescriptiveResult with ideal points and cutpoints.

    .. epigraph:: "Believe it!" -- Naruto Uzumaki, Naruto
    """
    from moirais._spatial_voting import ordinal_irt as _fn

    result = _fn(Y, n_dims=n_dims, n_samples=n_samples, burn_in=burn_in, seed=seed)
    return DescriptiveResult(
        name="ordinal_irt_model",
        value=result["n_samples"],
        extra=result,
    )


oirt = ordinal_irt_model


def cheatsheet() -> str:
    return "ordinal_irt_model({}) -> Ordinal IRT / mixed factor analysis."
