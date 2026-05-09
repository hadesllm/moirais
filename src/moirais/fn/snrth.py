"""SNR threshold for target BER."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your focus determines your reality."


def snr_threshold(snr_db, target_ber=1e-3, **kwargs) -> DescriptiveResult:
    """Evaluate whether SNR meets the target BER threshold.

    Uses the complementary error function relationship between
    SNR and BER for BPSK modulation.

    Parameters
    ----------
    snr_db : float
        Signal-to-noise ratio in decibels.
    target_ber : float
        Target bit error rate (default 1e-3).

    Returns
    -------
    DescriptiveResult
    """
    from scipy.special import erfc, erfcinv

    snr_linear = 10 ** (snr_db / 10.0)
    ber = 0.5 * erfc(np.sqrt(snr_linear))
    required_snr_linear = (erfcinv(2 * target_ber)) ** 2
    required_snr_db = 10.0 * np.log10(required_snr_linear)
    meets = bool(snr_db >= required_snr_db)
    return DescriptiveResult(
        name="snr_threshold",
        value=required_snr_db,
        extra={
            "snr_db": snr_db,
            "actual_ber": float(ber),
            "target_ber": target_ber,
            "required_snr_db": float(required_snr_db),
            "meets_threshold": meets,
        },
    )


snrth = snr_threshold


def cheatsheet() -> str:
    return "snr_threshold({}) -> SNR threshold for target BER."
