"""morie.schema — schema-agnostic column mapping.

A morie module names canonical concepts: "weight", "alcohol_past12m",
"heavy_drinking_30d", etc.  Users in the same field (sociolegal stats,
public health, criminology) frequently have datasets that hold those
concepts under different column names: `wt`, `drink_yn`, `binge30`, ...

This module lets those datasets flow through morie modules WITHOUT
forcing the user to rename their columns first.

Two paths are supported:

  1. **Explicit mapping** — pass ``column_mapping={"my_wt": "weight",
     "drinks_yn": "alcohol_past12m"}`` and the loader renames before
     handing to the canonical analysis layer.  Direction: user-column
     -> canonical-name.

  2. **Inferred mapping** — call :func:`infer_mapping(df,
     canonical=CPADS_REQUIRED_VARIABLES)` and morie attempts a
     fuzzy + dtype-aware match between the user's columns and the
     canonical names.  Confidence scores are returned so the caller
     can review before applying.

Both paths are intentionally simple.  The threshold for "matched"
defaults to 0.6 on the Levenshtein similarity ratio — high enough to
reject "id" -> "age_group" but low enough to accept "wt" -> "weight"
and "drink_yn" -> "alcohol_past12m".  Tune via the ``threshold``
parameter.

Example
-------

  >>> import pandas as pd
  >>> import morie.schema as ms
  >>> user_df = pd.DataFrame({
  ...     "wt": [1, 2, 3],
  ...     "drinks_yn": [0, 1, 1],
  ...     "binge30": [0, 0, 1],
  ...     "age_band": ["18-24", "25-34", "35-44"],
  ... })
  >>> mapping, scores = ms.infer_mapping(user_df,
  ...     canonical=["weight", "alcohol_past12m", "heavy_drinking_30d",
  ...                "age_group"])
  >>> mapping
  {'wt': 'weight', 'drinks_yn': 'alcohol_past12m', 'binge30':
   'heavy_drinking_30d', 'age_band': 'age_group'}
  >>> canon_df = ms.apply_mapping(user_df, mapping)
  >>> sorted(canon_df.columns)
  ['age_group', 'alcohol_past12m', 'heavy_drinking_30d', 'weight']
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable

import pandas as pd


# Synonyms that come up across criminology / public-health datasets.
# Used to nudge the fuzzy matcher when the literal column names diverge
# but the underlying concept is the same.
_SYNONYMS: dict[str, list[str]] = {
    "weight": [
        "wt", "wtpumf", "wtdf", "surveyweight", "design_weight", "pweight",
        "sampling_weight", "wgt",
    ],
    "alcohol_past12m": [
        "alc05", "alcohol", "drinker", "drinker_yn", "drinks_yn",
        "past_year_alcohol", "any_alcohol",
    ],
    "heavy_drinking_30d": [
        "binge30", "binge_30d", "heavy_binge", "hed", "alc12_30d_prev",
        "alc12_30d_prev_total", "binge_drink", "heavy_drinking",
    ],
    "ebac_tot": ["ebac", "bac_total", "blood_alcohol_total"],
    "ebac_legal": ["bac_legal", "blood_alcohol_legal", "above_legal_bac"],
    "cannabis_any_use": [
        "can05", "cannabis", "marijuana", "weed_yn", "any_cannabis",
    ],
    "age_group": [
        "age_band", "age_groups", "agegroup", "age_cat", "age_bin",
    ],
    "gender": [
        "sex", "dvdemq01", "gender_id", "sex_at_birth", "self_gender",
    ],
    "province_region": [
        "region", "province", "geo", "geo_region", "prov",
    ],
    "mental_health": [
        "hwbq02", "mh", "mh_self", "mental_self", "mh_status",
    ],
    "physical_health": [
        "hwbq01", "ph", "ph_self", "physical_self", "phys_status",
    ],
}


def _normalise(name: str) -> str:
    """Lowercase + strip non-alnum for fuzzy comparison."""
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _similarity(user_col: str, canonical: str) -> float:
    """Score 0-1 for how likely user_col maps to canonical.

    1. Exact (case-insensitive, punctuation-stripped) → 1.0
    2. Listed synonym → 0.95
    3. Substring containment → 0.80
    4. Levenshtein similarity → as-is
    """
    u = _normalise(user_col)
    c = _normalise(canonical)
    if u == c:
        return 1.0
    if user_col in _SYNONYMS.get(canonical, ()) or u in {_normalise(s) for s in _SYNONYMS.get(canonical, ())}:
        return 0.95
    if u in c or c in u:
        return 0.80
    return SequenceMatcher(None, u, c).ratio()


def infer_mapping(
    df: pd.DataFrame,
    *,
    canonical: Iterable[str],
    threshold: float = 0.6,
) -> tuple[dict[str, str], dict[str, float]]:
    """Infer a user-column → canonical-name mapping for one DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        User's input data.
    canonical : iterable of str
        The canonical names morie expects (e.g. CPADS_REQUIRED_VARIABLES).
    threshold : float, default 0.6
        Minimum similarity to accept a match.

    Returns
    -------
    mapping : dict[str, str]
        user_col -> canonical_name pairs above the threshold.
    scores : dict[str, float]
        The similarity score for each accepted pair.  Use this to
        decide whether to surface the mapping for user review.

    Notes
    -----
    Greedy: each user_col is assigned to its best-scoring canonical,
    and each canonical can only be matched once (first user_col
    wins).  Re-run with a tighter threshold for ambiguous datasets,
    or override with an explicit mapping.
    """
    canonical_list = list(canonical)
    user_cols = list(df.columns)
    pairs: list[tuple[float, str, str]] = []
    for u in user_cols:
        for c in canonical_list:
            s = _similarity(u, c)
            if s >= threshold:
                pairs.append((s, u, c))
    # Greedy assignment: highest scores first, no duplicates
    pairs.sort(reverse=True)
    used_user, used_canon = set(), set()
    mapping: dict[str, str] = {}
    scores: dict[str, float] = {}
    for s, u, c in pairs:
        if u in used_user or c in used_canon:
            continue
        mapping[u] = c
        scores[u] = s
        used_user.add(u)
        used_canon.add(c)
    return mapping, scores


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns in-order, return a new DataFrame.

    Columns not in the mapping are kept under their original name —
    morie modules tolerate extra columns; missing canonical names
    will be detected by `validate_cpads_frame` (or the equivalent)
    downstream.
    """
    return df.rename(columns=mapping)


def parse_cli_mapping(spec: str) -> dict[str, str]:
    """Parse a CLI-style mapping string into a dict.

    Format: ``my_wt:weight,drinks_yn:alcohol_past12m``.  Whitespace
    around items is tolerated.  Empty input yields an empty dict.
    """
    out: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"bad mapping entry {item!r}; expected user_col:canonical_name"
            )
        user, canon = item.split(":", 1)
        out[user.strip()] = canon.strip()
    return out
