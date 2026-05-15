# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mandela Rules apples-to-apples spectrum on OTIS b01.

Computes the full grid of provincial Mandela-classified rates across
four denominator conventions × three meaningful-contact proxies = 12
cells per fiscal year + a pooled row. Lets a downstream user pick
the row that matches their preferred operationalisation, and lets
the MRM paper Table 2 reproduce its spectrum from a single function
call.

Denominator conventions (per `mrm_classify_mandela` semantics):
  * row                  - per-placement rate (b01 row count)
  * individual_any       - share of within-year individuals with any
                           placement satisfying the criterion
  * individual_cumulative- share of within-year individuals whose
                           cumulative within-year segregation days
                           exceeds the threshold
  * c11_aggregate        - the duration-band aggregate from c11
                           (placeholder; requires the c11 file path)

Meaningful-contact proxies (Rule 44, derived from b01 alert flags):
  * none       - Rule 43 only, no contact proxy applied
  * any_alert  - Rule 43 ∩ (any of MentalHealth/SuicideRisk/SuicideWatch
                 alert active); these placements receive STAFF contact
                 from medical or watch staff, so this is the LOOSER
                 (less-restrictive) contact-failure proxy
  * no_alert   - Rule 43 ∩ (no alert active); these placements
                 receive NO recorded staff contact and are the
                 STRICTEST contact-failure proxy

The function returns a tidy long-format DataFrame with columns:
    year, denominator, contact_proxy, n_eligible, n_mandela, rate, pct
so the caller can pivot or filter at will.

Reference for the Rule 43 + Rule 44 definitions:
    United Nations General Assembly (2015). The Nelson Mandela Rules,
    A/RES/70/175.

Used by:
    - papers/mrm-formulations-paper §5.3 Table 2 (apples-to-apples
      spectrum row Federal SIU vs Ontario 2023/2024/2025)
    - papers/morie-empirical-paper §6 verification of all 12 cells.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


__all__ = ["mrm_otis_mandela_spectrum"]


CONTACT_PROXIES = ("none", "any_alert", "no_alert")
DENOMINATORS = ("row", "individual_any", "individual_cumulative", "c11_aggregate")
DEFAULT_ALERTS = ("MentalHealth_Alert", "SuicideRisk_Alert", "SuicideWatch_Alert")


def _is_yes(series: pd.Series) -> pd.Series:
    """Robust Yes/No coercion for OTIS alert columns."""
    return series.astype(str).str.strip().str.lower() == "yes"


def mrm_otis_mandela_spectrum(
    data: pd.DataFrame,
    *,
    duration_col: str = "NumberConsecutiveDays_Segregation",
    year_col: str = "EndFiscalYear",
    id_col: str = "UniqueIndividual_ID",
    threshold_days: int = 15,
    alert_cols: Iterable[str] = DEFAULT_ALERTS,
    contact_proxies: Iterable[str] = CONTACT_PROXIES,
    denominators: Iterable[str] = ("row", "individual_any", "individual_cumulative"),
    c11_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the full 4 × 3 Mandela spectrum on OTIS b01.

    Args:
        data: OTIS b01 frame (placement-level).
        duration_col, year_col, id_col: b01 column names.
        threshold_days: Rule 43 duration threshold (UN: 15 days).
        alert_cols: three b01 alert columns; "Yes" is treated as
            alert-active.
        contact_proxies: subset of {"none", "any_alert", "no_alert"}.
        denominators: subset of {"row", "individual_any",
            "individual_cumulative", "c11_aggregate"}. The
            "c11_aggregate" mode requires `c11_data` to be supplied
            and is skipped silently otherwise.
        c11_data: optional c11 aggregate frame. When supplied, the
            c11_aggregate denominator joins per-individual cumulative
            duration from c11.

    Returns:
        Tidy DataFrame: one row per (year, denominator, contact_proxy)
        combination plus the pooled row, with columns
            year, denominator, contact_proxy, n_eligible, n_mandela,
            rate, pct.

    Examples:
        >>> b01 = pd.read_csv("b01_segregation_detailed_dataset.csv")
        >>> spectrum = mrm_otis_mandela_spectrum(b01)
        >>> spectrum.pivot_table(index=["year", "denominator"],
        ...                     columns="contact_proxy",
        ...                     values="pct")
    """
    df = data.copy()
    dur = pd.to_numeric(df[duration_col], errors="coerce")
    df["_dur"] = dur
    df["_long"] = (dur > threshold_days).fillna(False)

    # Build alert masks
    df["_any_alert"] = False
    df["_no_alert"] = True
    for c in alert_cols:
        if c not in df.columns:
            continue
        yes = _is_yes(df[c])
        df["_any_alert"] |= yes
        df["_no_alert"] &= ~yes

    # Per (year, denominator, proxy) computation
    years = sorted(pd.unique(df[year_col].dropna()))
    rows: list[dict] = []

    def _eligible_mask(proxy: str) -> pd.Series:
        if proxy == "none":
            return df["_long"]
        if proxy == "any_alert":
            return df["_long"] & df["_any_alert"]
        if proxy == "no_alert":
            return df["_long"] & df["_no_alert"]
        raise ValueError(f"unknown contact_proxy {proxy!r}")

    for y in [*years, "pooled"]:
        if y == "pooled":
            ymask = pd.Series(True, index=df.index)
            label = "pooled"
        else:
            ymask = df[year_col] == y
            label = str(int(y)) if not isinstance(y, str) else y

        for proxy in contact_proxies:
            elig = _eligible_mask(proxy) & ymask

            for denom in denominators:
                if denom == "row":
                    n_d = int(ymask.sum())
                    n_m = int(elig.sum())
                elif denom == "individual_any":
                    ids = df.loc[ymask, id_col].dropna().unique()
                    ids_m = df.loc[elig, id_col].dropna().unique()
                    n_d = int(ids.size)
                    n_m = int(ids_m.size)
                elif denom == "individual_cumulative":
                    sub = df.loc[ymask]
                    cum = sub.groupby(id_col)["_dur"].sum()
                    ids_m = (
                        df.loc[elig]
                          .groupby(id_col)["_dur"].sum()
                    )
                    cum_long = cum > threshold_days
                    # restrict cum_long to ids that had alert-proxy-active placements
                    cum_long_proxy = cum_long.index.isin(ids_m.index) & cum_long.values
                    n_d = int(cum.size)
                    if proxy == "none":
                        n_m = int(cum_long.sum())
                    else:
                        n_m = int(np.array(cum_long_proxy, dtype=bool).sum())
                elif denom == "c11_aggregate":
                    if c11_data is None:
                        continue
                    # c11 has per-individual cumulative duration bands;
                    # we sum NumberIndividuals_Segregation for bands > threshold
                    if y == "pooled":
                        sub = c11_data
                    else:
                        sub = c11_data[c11_data[year_col] == int(y) if isinstance(y, str) else y]
                    n_d_col = "NumberIndividuals_Segregation"
                    if n_d_col not in sub.columns:
                        continue
                    n_d = int(sub[n_d_col].sum())
                    # The "Aggregate_Duration" string e.g. "21 to 25 days" or "Greater than 30 days"
                    durband_col = "Aggregate_Duration"
                    if durband_col not in sub.columns:
                        continue
                    above = sub[durband_col].astype(str).map(
                        lambda b: ("Greater than" in b) or any(
                            int(x) > threshold_days
                            for x in __import__("re").findall(r"\d+", b)
                            if int(x) > threshold_days
                        )
                    )
                    n_m = int(sub.loc[above, n_d_col].sum())
                else:
                    raise ValueError(f"unknown denominator {denom!r}")

                rate = (n_m / n_d) if n_d > 0 else float("nan")
                rows.append({
                    "year": label,
                    "denominator": denom,
                    "contact_proxy": proxy,
                    "n_eligible": n_d,
                    "n_mandela": n_m,
                    "rate": round(rate, 6) if not np.isnan(rate) else rate,
                    "pct": round(100 * rate, 2) if not np.isnan(rate) else rate,
                })

    return pd.DataFrame(rows)
