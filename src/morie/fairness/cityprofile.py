# SPDX-License-Identifier: AGPL-3.0-or-later
"""City-agnostic data profiles for the predictive-policing audit.

The disparity audit in :mod:`morie.fairness.predpol` operates on a
*canonical* per-area schema — ``area``, ``risk``, ``outcome``,
``population``, ``group`` — so the same audit code runs for Chicago,
New York, Toronto, or any other city.

A :class:`CityProfile` records which columns in one city's open-data
export carry those five canonical fields.  :func:`apply_profile`
renames an arbitrary city ``DataFrame`` onto the canonical schema; the
audit then never has to know which city the data came from.

This is the generalisation step the SciencesPo
*Predictive-policing-Chicago* project (Lachérade, Szabo, Krikava &
Aeby, 2021) did not need — that study was Chicago-only.  morie keeps
the *method* and drops the city assumption.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CANONICAL_FIELDS",
    "CityProfile",
    "register_city",
    "get_city",
    "list_cities",
    "apply_profile",
]

#: The five canonical per-area fields the audit consumes.
CANONICAL_FIELDS = ("area", "risk", "outcome", "population", "group")


@dataclass(frozen=True)
class CityProfile:
    """Column mapping for one city's predictive-policing data.

    Each ``*_col`` names the column, in that city's own export, that
    carries the corresponding canonical field.  ``risk_col`` and
    ``outcome_col`` may be ``None`` when a city only supplies one side
    (e.g. risk scores but no realised-outcome feed), in which case the
    missing side must be supplied separately to the audit.

    Parameters
    ----------
    name : str
        Identifier used with :func:`get_city` (e.g. ``"chicago"``).
    area_col : str
        Column holding the area / district / precinct identifier.
    risk_col, outcome_col, population_col, group_col : str or None
        Columns for the predicted risk score, realised-outcome count,
        area population, and protected attribute.
    notes : str
        Free-text provenance / caveats, surfaced to the user.
    """

    name: str
    area_col: str
    risk_col: str | None = None
    outcome_col: str | None = None
    population_col: str | None = None
    group_col: str | None = None
    notes: str = ""

    def column_map(self) -> dict[str, str]:
        """Return ``{source_column: canonical_field}`` for the columns
        this profile actually defines."""
        pairs = {
            self.area_col: "area",
            self.risk_col: "risk",
            self.outcome_col: "outcome",
            self.population_col: "population",
            self.group_col: "group",
        }
        return {src: canon for src, canon in pairs.items() if src is not None}


# The "generic" profile: the DataFrame is already in canonical form.
# Cities are registered by users (or by morie's ingest wiring) with
# their own column names — see register_city().
_REGISTRY: dict[str, CityProfile] = {
    "generic": CityProfile(
        name="generic",
        area_col="area",
        risk_col="risk",
        outcome_col="outcome",
        population_col="population",
        group_col="group",
        notes="Identity profile — the DataFrame already uses the "
              "canonical column names.",
    ),
}


def register_city(profile: CityProfile, *, overwrite: bool = False) -> None:
    """Add (or replace) a :class:`CityProfile` in the registry.

    Parameters
    ----------
    profile : CityProfile
    overwrite : bool
        If ``False`` (default) registering an existing name raises
        ``ValueError``; pass ``True`` to replace it.
    """
    key = profile.name.strip().lower()
    if key in _REGISTRY and not overwrite:
        raise ValueError(
            f"city {key!r} is already registered; pass overwrite=True "
            f"to replace it"
        )
    _REGISTRY[key] = profile


def get_city(name: str) -> CityProfile:
    """Look up a registered :class:`CityProfile` by name (case-insensitive)."""
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"no city profile {name!r}; registered: {sorted(_REGISTRY)}. "
            f"Register one with morie.fairness.register_city(...)."
        )
    return _REGISTRY[key]


def list_cities() -> list[str]:
    """Return the names of all registered city profiles, sorted."""
    return sorted(_REGISTRY)


def apply_profile(df, profile: CityProfile | str):
    """Rename a city ``DataFrame`` onto the canonical audit schema.

    Parameters
    ----------
    df : pandas.DataFrame
        A city's predictive-policing data in its native column names.
    profile : CityProfile or str
        The profile (or the name of a registered profile) describing
        ``df``'s columns.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the profile's columns renamed to the
        canonical names and only those canonical columns retained.

    Raises
    ------
    KeyError
        If a column the profile names is absent from ``df``.
    """
    if isinstance(profile, str):
        profile = get_city(profile)
    colmap = profile.column_map()
    missing = [c for c in colmap if c not in df.columns]
    if missing:
        raise KeyError(
            f"profile {profile.name!r} expects column(s) {missing} which "
            f"are not in the DataFrame (columns: {list(df.columns)})"
        )
    out = df[list(colmap)].rename(columns=colmap).copy()
    return out
