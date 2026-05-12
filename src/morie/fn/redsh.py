# morie.fn — function file (hadesllm/morie)
"""Cosmological redshift."""

__all__ = ["redsh"]



def redsh(
    z: float = None,
    wavelength_obs: float = None,
    wavelength_emit: float = None,
    a_emit: float = None,
    a_obs: float = 1.0,
) -> dict:
    r"""
    Compute cosmological redshift and related quantities.

    .. math::

        1 + z = \\frac{\\lambda_{\\text{obs}}}{\\lambda_{\\text{emit}}}
              = \\frac{a(t_{\\text{obs}})}{a(t_{\\text{emit}})}

    Provide either z, or (wavelength_obs, wavelength_emit), or a_emit.

    Parameters
    ----------
    z : float, optional
        Known redshift.
    wavelength_obs : float, optional
        Observed wavelength.
    wavelength_emit : float, optional
        Emitted (rest-frame) wavelength.
    a_emit : float, optional
        Scale factor at emission.
    a_obs : float
        Scale factor at observation (default 1.0, today).

    Returns
    -------
    dict
        Keys: z, scale_factor_ratio, velocity_approx_km_s,
        velocity_relativistic_km_s.
    """
    c_km_s = 299792.458

    if z is not None:
        pass
    elif wavelength_obs is not None and wavelength_emit is not None:
        if wavelength_emit <= 0:
            raise ValueError("wavelength_emit must be > 0.")
        z = (wavelength_obs - wavelength_emit) / wavelength_emit
    elif a_emit is not None:
        if a_emit <= 0:
            raise ValueError("a_emit must be > 0.")
        z = a_obs / a_emit - 1.0
    else:
        raise ValueError("Provide z, (wavelength_obs, wavelength_emit), or a_emit.")

    if z < -1.0:
        raise ValueError("z must be > -1.")

    v_approx = z * c_km_s

    zp1 = 1.0 + z
    beta = (zp1 ** 2 - 1.0) / (zp1 ** 2 + 1.0)
    v_rel = beta * c_km_s

    return {
        "z": float(z),
        "scale_factor_ratio": float(1.0 + z),
        "velocity_approx_km_s": float(v_approx),
        "velocity_relativistic_km_s": float(v_rel),
    }
