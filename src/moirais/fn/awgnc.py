# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""AWGN channel capacity (Shannon limit)."""

__all__ = ["awgnc"]

import numpy as np
from ._richresult import RichResult


def awgnc(snr_db: float = None, *, snr_linear: float = None) -> dict:
    """
    Capacity of an additive white Gaussian noise channel.

    .. math::

        C = \\frac{1}{2} \\log_2(1 + \\text{SNR})

    Parameters
    ----------
    snr_db : float, optional
        Signal-to-noise ratio in dB.
    snr_linear : float, optional
        Signal-to-noise ratio in linear scale.
        Exactly one of snr_db or snr_linear must be provided.

    Returns
    -------
    dict
        'capacity' (bits/channel use), 'snr_db', 'snr_linear'.

    Raises
    ------
    ValueError
        If neither or both SNR parameters provided, or SNR < 0.

    References
    ----------
    Shannon, C. E. (1948). A mathematical theory of communication.
    Bell System Tech. J., 27, 379-423, 623-656.
    """
    if snr_db is not None and snr_linear is not None:
        raise ValueError("Provide only one of snr_db or snr_linear.")
    if snr_db is None and snr_linear is None:
        raise ValueError("Must provide snr_db or snr_linear.")
    if snr_linear is not None:
        if snr_linear < 0:
            raise ValueError("snr_linear must be >= 0.")
        snr_lin = float(snr_linear)
        snr_d = 10.0 * np.log10(snr_lin) if snr_lin > 0 else -np.inf
    else:
        snr_d = float(snr_db)
        snr_lin = 10.0 ** (snr_d / 10.0)

    cap = 0.5 * np.log2(1.0 + snr_lin)
    return RichResult(payload={"capacity": cap, "snr_db": snr_d, "snr_linear": snr_lin})
