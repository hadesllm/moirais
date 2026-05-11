# morie.fn — function file (hadesllm/morie)
"""Impact testing energy (Charpy/Izod). 'I want everything.' -- Bane (Smith)"""

from __future__ import annotations

from ._containers import DescriptiveResult


def impact_energy(
    initial_height: float,
    final_height: float,
    *,
    pendulum_mass: float = 25.0,
    g: float = 9.81,
    specimen_area: float | None = None,
    method: str = "charpy",
) -> DescriptiveResult:
    """Compute absorbed energy from a Charpy or Izod impact test.

    Energy_absorbed = m * g * (h_initial - h_final)

    Parameters
    ----------
    initial_height : float
        Height of pendulum before release (metres).
    final_height : float
        Height of pendulum after impact (metres).
    pendulum_mass : float
        Mass of the pendulum striker (kg).
    g : float
        Gravitational acceleration (m/s^2).
    specimen_area : float or None
        Cross-sectional area behind the notch (m^2). If given, also
        computes impact strength (energy / area).
    method : str
        ``"charpy"`` or ``"izod"``.

    Returns
    -------
    DescriptiveResult
        ``value`` is the absorbed energy in Joules.
    """
    if initial_height < 0 or final_height < 0:
        raise ValueError("Heights must be non-negative")
    if final_height > initial_height:
        raise ValueError("final_height cannot exceed initial_height")
    if pendulum_mass <= 0:
        raise ValueError("pendulum_mass must be positive")
    if method not in ("charpy", "izod"):
        raise ValueError(f"method must be 'charpy' or 'izod', got '{method}'")

    energy = pendulum_mass * g * (initial_height - final_height)
    extra = {
        "method": method,
        "pendulum_mass_kg": pendulum_mass,
        "h_initial_m": initial_height,
        "h_final_m": final_height,
        "g": g,
    }
    if specimen_area is not None:
        if specimen_area <= 0:
            raise ValueError("specimen_area must be positive")
        impact_strength = energy / specimen_area
        extra["specimen_area_m2"] = specimen_area
        extra["impact_strength_J_m2"] = float(impact_strength)

    return DescriptiveResult(
        name=f"Impact Energy ({method.title()})",
        value=float(energy),
        extra=extra,
    )


hamrm = impact_energy


def cheatsheet() -> str:
    return "impact_energy({}) -> Impact testing energy (Charpy/Izod). 'I want everything.' --"
