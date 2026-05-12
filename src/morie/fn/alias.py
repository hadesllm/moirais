# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Aliasing demonstration."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "I've got a bad feeling about this."


def aliasing_demo(f_signal: float, fs: float) -> DescriptiveResult:
    r"""Show the aliased frequency when fs < 2 * f_signal.

    .. math::

        f_{\\text{alias}} = |f_{\\text{signal}} - k \\cdot f_s|,
        \\quad k = \\text{round}(f_{\\text{signal}} / f_s)

    Parameters
    ----------
    f_signal : float
        True signal frequency (Hz).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    DescriptiveResult
    """
    nyquist = fs / 2.0
    aliased = fs > 0
    f_alias = abs(f_signal % fs)
    if f_alias > nyquist:
        f_alias = fs - f_alias
    is_aliased = f_signal > nyquist
    return DescriptiveResult(
        name="aliasing_demo",
        value=f_alias,
        extra={
            "f_signal": f_signal,
            "fs": fs,
            "nyquist": nyquist,
            "f_alias": f_alias,
            "is_aliased": is_aliased,
        },
    )


alias = aliasing_demo


def cheatsheet() -> str:
    return "aliasing_demo({}) -> Aliasing demonstration."
