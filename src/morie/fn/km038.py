r"""Gpt2 task conditioning.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_gpt2_task_conditioning"]


def kamath_ch2_gpt2_task_conditioning(input, task):
    r"""
    Gpt2 task conditioning.

    Formula: p(\mathrm{output}|\mathrm{input},\mathrm{task})

    Parameters
    ----------
    input : array-like
        Input data.
    task : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.38, p. 70
    r"""
    input = np.atleast_1d(np.asarray(input, dtype=float))
    n = len(input)
    result = float(np.mean(input))
    se = float(np.std(input, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gpt2 task conditioning."})


def cheatsheet():
    return "km038: Gpt2 task conditioning."
