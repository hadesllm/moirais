"""Bundled Solar System demo dataset (public-domain astronomical facts)."""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult

_PLANETS = pd.DataFrame(
    {
        "name": ["Mercury", "Venus", "Earth", "Mars", "Jupiter",
                 "Saturn", "Uranus", "Neptune"],
        "mass_earths": [0.055, 0.815, 1.0, 0.107, 317.8,
                        95.2, 14.5, 17.1],
        "radius_km": [2440, 6052, 6371, 3390, 69911,
                      58232, 25362, 24622],
        "moons": [0, 0, 1, 2, 95, 146, 28, 16],
        "orbital_period_days": [88.0, 224.7, 365.25, 687.0, 4332.6,
                                10759.2, 30688.5, 60182.0],
    }
)

_MOONS = pd.DataFrame(
    {
        "name": ["Moon", "Phobos", "Deimos", "Io", "Europa",
                 "Ganymede", "Callisto", "Titan", "Enceladus"],
        "planet": ["Earth", "Mars", "Mars", "Jupiter", "Jupiter",
                   "Jupiter", "Jupiter", "Saturn", "Saturn"],
        "radius_km": [1737, 11, 6, 1822, 1561,
                      2634, 2410, 2575, 252],
    }
)

_MISSIONS = pd.DataFrame(
    {
        "name": ["Sputnik 1", "Apollo 11", "Voyager 1", "Voyager 2",
                 "Cassini", "Curiosity", "New Horizons"],
        "launch_year": [1957, 1969, 1977, 1977, 1997, 2011, 2006],
        "target": ["Earth orbit", "Moon", "outer planets",
                   "outer planets", "Saturn", "Mars", "Pluto"],
    }
)

_TABLES = {
    "planets": _PLANETS,
    "moons": _MOONS,
    "missions": _MISSIONS,
}


def load_solar_system(table: str = "planets") -> pd.DataFrame:
    """Load a table from the bundled Solar System demo dataset.

    Parameters
    ----------
    table : str, default "planets"
        One of: planets, moons, missions.

    Returns
    -------
    pd.DataFrame
    """
    key = table.lower().strip()
    if key not in _TABLES:
        raise ValueError(
            f"Unknown table '{table}'. Choose from: {sorted(_TABLES)}")
    return _TABLES[key].copy()


def load_solar_system_result(table: str = "planets") -> DescriptiveResult:
    """Load a Solar System table and wrap it in a DescriptiveResult."""
    df = load_solar_system(table)
    return DescriptiveResult(name=f"Solar System: {table}", value=df)


swdb = load_solar_system_result


def cheatsheet() -> str:
    return "load_solar_system({}) -> Solar System demo dataset loader."
