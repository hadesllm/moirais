# morie.fn -- function file (hadesllm/morie)
"""Letter frequency analysis for substitution cipher detection."""

from __future__ import annotations

from ._containers import DescriptiveResult


def cipher_frequency(
    text: str,
    *,
    reference: dict[str, float] | None = None,
) -> DescriptiveResult:
    """Letter frequency analysis for substitution cipher detection.

    Compares the character frequency distribution of *text* against a reference
    distribution (default: English) using chi-squared divergence.  Useful in
    cryptanalysis, stylometry, and authorship attribution.

    Parameters
    ----------
    text : str
        Input text to analyse.
    reference : dict or None
        Expected letter frequencies {char: proportion}.  If None, uses
        standard English letter frequencies.

    Returns
    -------
    DescriptiveResult
        ``value`` = chi-squared statistic comparing observed vs expected.
    """
    if not isinstance(text, str) or len(text) == 0:
        raise ValueError("text must be a non-empty string")
    if reference is None:
        reference = {
            "a": 0.0817,
            "b": 0.0150,
            "c": 0.0278,
            "d": 0.0425,
            "e": 0.1270,
            "f": 0.0223,
            "g": 0.0202,
            "h": 0.0609,
            "i": 0.0697,
            "j": 0.0015,
            "k": 0.0077,
            "l": 0.0403,
            "m": 0.0241,
            "n": 0.0675,
            "o": 0.0751,
            "p": 0.0193,
            "q": 0.0010,
            "r": 0.0599,
            "s": 0.0633,
            "t": 0.0906,
            "u": 0.0276,
            "v": 0.0098,
            "w": 0.0236,
            "x": 0.0015,
            "y": 0.0197,
            "z": 0.0007,
        }
    letters = [c.lower() for c in text if c.isalpha()]
    n = len(letters)
    if n == 0:
        raise ValueError("No alphabetic characters in text")
    observed: dict[str, int] = {}
    for ch in letters:
        observed[ch] = observed.get(ch, 0) + 1
    chi2 = 0.0
    freq_table: dict[str, dict[str, float]] = {}
    for ch, expected_prop in reference.items():
        obs_count = observed.get(ch, 0)
        exp_count = expected_prop * n
        if exp_count > 0:
            chi2 += (obs_count - exp_count) ** 2 / exp_count
        freq_table[ch] = {
            "observed": obs_count / n,
            "expected": expected_prop,
            "diff": obs_count / n - expected_prop,
        }
    ic = sum(c * (c - 1) for c in observed.values()) / (n * (n - 1)) if n > 1 else 0.0
    return DescriptiveResult(
        name="Cipher frequency analysis",
        value=float(chi2),
        extra={
            "n_letters": n,
            "n_unique": len(observed),
            "index_of_coincidence": round(ic, 6),
            "english_ic": 0.0667,
            "top5": sorted(observed.items(), key=lambda kv: -kv[1])[:5],
            "chi2": float(chi2),
        },
    )


cipfre = cipher_frequency


def cheatsheet() -> str:
    return 'cipher_frequency({}) -> Enigma cipher / substitution cipher analysis.'
