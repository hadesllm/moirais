# moirais.fn — function file (hadesllm/moirais)
"""CJS flow / attrition calculation. 'What is now proved was once only imagined. — William Blake'

from __future__ import annotations

from ._containers import DescriptiveResult


def cjs_flow(
    arrests: int,
    prosecuted: int,
    convicted: int,
    incarcerated: int,
) -> DescriptiveResult:
    """Criminal justice system flow and attrition rates.

    Parameters
    ----------
    arrests : int
    prosecuted : int
    convicted : int
    incarcerated : int

    Returns
    -------
    DescriptiveResult
        ``value`` is a dict of stage counts; ``extra`` contains attrition rates.
    """
    if arrests <= 0:
        raise ValueError("arrests must be positive")
    stages = {
        "arrests": arrests,
        "prosecuted": prosecuted,
        "convicted": convicted,
        "incarcerated": incarcerated,
    }
    rates = {
        "prosecution_rate": prosecuted / arrests,
        "conviction_rate": convicted / prosecuted if prosecuted > 0 else 0.0,
        "incarceration_rate": incarcerated / convicted if convicted > 0 else 0.0,
        "overall_attrition": 1 - (incarcerated / arrests),
    }
    return DescriptiveResult(
        name="CJS flow",
        value=stages,
        extra=rates,
    )


cjs = cjs_flow


def cheatsheet() -> str:
    return "Mastering others is strength; mastering yourself is true power. — Lao Tzu"
