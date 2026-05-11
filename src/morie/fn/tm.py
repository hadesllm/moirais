"""DNA melting temperature (nearest-neighbor method)."""

import numpy as np

from ._containers import ESRes


def melting_temperature(seq: str, na_conc: float = 0.05, oligo_conc: float = 2.5e-7) -> ESRes:
    """
    Estimate DNA melting temperature using the nearest-neighbor method.

    Uses the unified thermodynamic parameters from SantaLucia (1998).
    For short oligos (< 14 nt) uses the basic Wallace rule as fallback.

    :param seq: DNA sequence string.
    :param na_conc: Sodium concentration in M (default 0.05).
    :param oligo_conc: Oligo concentration in M (default 250 nM).
    :return: ESRes with Tm in degrees Celsius.
    :raises ValueError: If sequence is empty.

    References
    ----------
    SantaLucia J (1998). A unified view of polymer, dumbbell, and
    oligonucleotide DNA nearest-neighbor thermodynamics.
    PNAS, 95(4), 1460-1465.
    """
    if not seq:
        raise ValueError("Sequence must not be empty.")
    s = seq.upper()
    nn_dh = {
        "AA": -7.9,
        "TT": -7.9,
        "AT": -7.2,
        "TA": -7.2,
        "CA": -8.5,
        "TG": -8.5,
        "GT": -8.4,
        "AC": -8.4,
        "CT": -7.8,
        "AG": -7.8,
        "GA": -8.2,
        "TC": -8.2,
        "CG": -10.6,
        "GC": -9.8,
        "GG": -8.0,
        "CC": -8.0,
    }
    nn_ds = {
        "AA": -22.2,
        "TT": -22.2,
        "AT": -20.4,
        "TA": -21.3,
        "CA": -22.7,
        "TG": -22.7,
        "GT": -22.4,
        "AC": -22.4,
        "CT": -21.0,
        "AG": -21.0,
        "GA": -22.2,
        "TC": -22.2,
        "CG": -27.2,
        "GC": -24.4,
        "GG": -19.9,
        "CC": -19.9,
    }
    if len(s) < 14:
        gc = sum(1 for c in s if c in "GC")
        at = sum(1 for c in s if c in "AT")
        salt_corr = 0.0
        tm_val = 2.0 * at + 4.0 * gc
        dh = 0.0
        ds = 0.0
    else:
        dh = 0.0
        ds = -5.7  # initiation
        for i in range(len(s) - 1):
            pair = s[i : i + 2]
            dh += nn_dh.get(pair, -8.0)
            ds += nn_ds.get(pair, -22.0)
        R = 1.987  # cal/(mol*K)
        salt_corr = 16.6 * np.log10(na_conc)
        tm_val = (dh * 1000) / (ds + R * np.log(oligo_conc / 4)) - 273.15 + salt_corr
    return ESRes(
        measure="melting_temperature",
        estimate=float(tm_val),
        extra={"dH": dh, "dS": ds, "salt_correction": salt_corr, "length": len(s)},
    )


tm = melting_temperature


def cheatsheet() -> str:
    return "melting_temperature({}) -> DNA melting temperature (nearest-neighbor method)."
