# moirais.fn — function file (hadesllm/moirais)
"""Effective number of bits."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "This is the way."


def enob_compute(sinad_db: float, **kwargs) -> DescriptiveResult:
    """Compute the effective number of bits from SINAD.

    .. math::

        \\text{ENOB} = \\frac{\\text{SINAD}_{\\text{dB}} - 1.76}{6.02}

    Parameters
    ----------
    sinad_db : float
        SINAD value in dB.

    Returns
    -------
    DescriptiveResult
    """
    enob_val = (sinad_db - 1.76) / 6.02
    return DescriptiveResult(
        name="enob",
        value=enob_val,
        extra={"sinad_db": sinad_db, "enob": enob_val},
    )


enob = enob_compute


def cheatsheet() -> str:
    return "enob_compute({}) -> Effective number of bits."
