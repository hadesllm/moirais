# morie.fn -- function file (hadesllm/morie)
"""Deviance Information Criterion. 'Powerful you have become. -- Dooku'"""

from __future__ import annotations

from ._containers import DescriptiveResult


def deviance_info_criterion(
    D_bar: float,
    D_hat: float,
) -> DescriptiveResult:
    """Deviance Information Criterion (Spiegelhalter et al., 2002).

    DIC = D_bar + p_D  where  p_D = D_bar - D_hat
    is the effective number of parameters.

    :param D_bar: Posterior mean deviance E[-2 log p(y|theta)].
    :param D_hat: Deviance at the posterior mean of theta.
    :return: DescriptiveResult with DIC value.
    """
    p_D = D_bar - D_hat
    dic_val = D_bar + p_D
    return DescriptiveResult(
        name="dic",
        value=float(dic_val),
        extra={"D_bar": D_bar, "D_hat": D_hat, "p_D": p_D},
    )


dic = deviance_info_criterion


def cheatsheet() -> str:
    return "deviance_info_criterion({}) -> Deviance Information Criterion. 'Powerful you have become. -"
