# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayes theorem."""


from ._containers import ESRes

_QUOTE = "Everything flows. — Heraclitus"


def bayes_theorem(
    prior: float,
    likelihood: float,
    evidence: float | None = None,
    **kwargs,
) -> ESRes:
    r"""
    Apply Bayes' theorem to compute posterior probability.

    .. math::

        P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}

    If *evidence* is ``None`` it is computed assuming a binary hypothesis
    space: P(B) = P(B|A)P(A) + P(B|~A)P(~A), requiring *likelihood_h0*
    in ``kwargs``.

    :param prior: P(A), prior probability.
    :param likelihood: P(B|A), likelihood.
    :param evidence: P(B), marginal likelihood / evidence.  Optional.
    :param kwargs: Pass ``likelihood_h0`` for the alternative hypothesis.
    :return: ESRes with posterior probability.

    References
    ----------
    Bayes T (1763). An essay towards solving a problem in the
    doctrine of chances. Philosophical Transactions, 53, 370-418.
    """
    if not 0 <= prior <= 1:
        raise ValueError("prior must be in [0, 1].")
    if not 0 <= likelihood <= 1:
        raise ValueError("likelihood must be in [0, 1].")
    if evidence is None:
        lh0 = kwargs.get("likelihood_h0")
        if lh0 is None:
            raise ValueError("Provide evidence or likelihood_h0 for binary complement.")
        evidence = likelihood * prior + lh0 * (1 - prior)
    if evidence <= 0:
        raise ValueError("evidence must be > 0.")
    posterior = (likelihood * prior) / evidence
    return ESRes(
        measure="bayes_theorem",
        estimate=float(posterior),
        extra={
            "prior": prior,
            "likelihood": likelihood,
            "evidence": evidence,
        },
    )


bayes = bayes_theorem


def cheatsheet() -> str:
    return "bayes_theorem({}) -> Bayes theorem."
