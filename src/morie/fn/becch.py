from ._richresult import RichResult
# morie.fn — function file (hadesllm/morie)
"""Binary erasure channel capacity."""

__all__ = ["becch"]


def becch(epsilon: float) -> dict:
    r"""
    Capacity of a binary erasure channel (BEC).

    .. math::

        C = 1 - \\epsilon

    Parameters
    ----------
    epsilon : float
        Erasure probability, 0 <= epsilon <= 1.

    Returns
    -------
    dict
        'capacity' (bits), 'erasure_prob'.

    Raises
    ------
    ValueError
        If epsilon not in [0, 1].

    References
    ----------
    Cover & Thomas (2006). Elements of Information Theory, Ch. 7.
    """
    if not 0 <= epsilon <= 1:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}.")
    return RichResult(payload={"capacity": 1.0 - epsilon, "erasure_prob": epsilon})
