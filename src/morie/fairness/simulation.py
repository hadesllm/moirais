# SPDX-License-Identifier: AGPL-3.0-or-later
"""Simulation primitives for the predictive-policing audit subsystem.

Clean-room reimplementations of the simulation components in Barman &
Barman, *Unmasking Algorithmic Bias in Predictive Policing*
(arXiv:2603.18987):

* :func:`noisy_or_detection` — the Noisy-OR patrol-detection model: the
  probability that a crime is detected, given the patrol officers near
  it.
* :func:`simulate_biased_crime_data` — a synthetic predictive-policing
  dataset generator with a *known, controllable* injected disparity,
  for exercising and verifying the :mod:`morie.fairness` audits.

Both are pure-numpy and need no optional dependency.  The JAX GAN /
CTGAN generative models live in :mod:`morie.fairness.gan` (the
``morie[sim]`` extra).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from morie.fn._richresult import RichResult

__all__ = ["noisy_or_detection", "simulate_biased_crime_data"]


def noisy_or_detection(
    crime_xy: Any,
    officer_xy: Any,
    *,
    radius: float,
    p_detect: float = 0.85,
    seed: int | None = None,
) -> RichResult:
    """Noisy-OR detection probability for each crime given patrol officers.

    A crime is detected if *any* nearby officer detects it.  With a
    per-officer detection probability ``p`` and ``k`` officers within
    ``radius``::

        P(detected | crime) = 1 − ∏ (1 − p) = 1 − (1 − p) ** k

    Parameters
    ----------
    crime_xy : array-like, shape (n, 2)
        Crime-event coordinates.
    officer_xy : array-like, shape (m, 2)
        Patrol-officer coordinates.
    radius : float
        Detection radius (same units as the coordinates).
    p_detect : float, default 0.85
        Per-officer detection probability, in ``(0, 1]``.
    seed : int, optional
        If given, a 0/1 detection outcome is also sampled.

    Returns
    -------
    RichResult
        Headline value is the mean per-crime detection probability;
        payload carries ``probabilities``, ``officers_in_range`` and,
        when ``seed`` is set, the sampled ``detected`` vector.

    Examples
    --------
    >>> import morie
    >>> crimes = [[0.0, 0.0], [10.0, 10.0]]
    >>> officers = [[0.1, 0.1]]            # near the first crime only
    >>> res = morie.noisy_or_detection(crimes, officers, radius=1.0)
    >>> round(float(res.payload["probabilities"][0]), 2)
    0.85
    >>> float(res.payload["probabilities"][1])    # no officer in range
    0.0
    """
    crime = np.asarray(crime_xy, dtype=float)
    officer = np.asarray(officer_xy, dtype=float)
    if crime.ndim != 2 or crime.shape[1] != 2:
        raise ValueError("crime_xy must be an (n, 2) array of coordinates")
    if officer.ndim != 2 or officer.shape[1] != 2:
        raise ValueError("officer_xy must be an (m, 2) array of coordinates")
    if not (0.0 < p_detect <= 1.0):
        raise ValueError("p_detect must be in (0, 1]")
    if radius <= 0:
        raise ValueError("radius must be positive")

    if officer.shape[0] == 0:
        k = np.zeros(crime.shape[0], dtype=int)
    else:
        d = np.sqrt(((crime[:, None, :] - officer[None, :, :]) ** 2).sum(-1))
        k = (d <= radius).sum(axis=1)
    prob = 1.0 - (1.0 - p_detect) ** k

    detected = None
    if seed is not None:
        rng = np.random.default_rng(seed)
        detected = (rng.random(prob.shape[0]) < prob).astype(int)

    mean_p = float(np.mean(prob)) if prob.size else float("nan")
    interp = (
        f"Across {crime.shape[0]} crime events and {officer.shape[0]} "
        f"patrol officers, the mean detection probability is {mean_p:.3f}. "
        f"{int(np.sum(k == 0))} crime(s) had no officer within radius "
        f"(detection probability 0)."
    )
    return RichResult(
        title="Noisy-OR Patrol Detection",
        summary_lines=[
            ("Crimes", int(crime.shape[0])),
            ("Officers", int(officer.shape[0])),
            ("Mean detection probability", mean_p),
            ("Per-officer probability", p_detect),
        ],
        interpretation=interp,
        payload={
            "value": mean_p,
            "probabilities": prob,
            "officers_in_range": k,
            "detected": detected,
        },
    )


def simulate_biased_crime_data(
    *,
    n: int = 2000,
    groups: Any = ("A", "B"),
    group_props: Any = None,
    n_areas: int = 20,
    base_rate: float = 0.3,
    bias: float = 0.5,
    seed: int = 0,
):
    """Generate a synthetic predictive-policing dataset with known bias.

    Every record carries an ``area`` (areas are group-segregated, as in
    Chicago), a ``group``, a group-*independent* ``true_outcome``, the
    system's ``detected`` flag, and a continuous ``risk_score`` (0–500,
    SSL-style).  The ``bias`` parameter is the ground truth the audits
    should recover:

    * ``bias = 0`` — perfectly fair: every group is flagged at
      ``base_rate``.
    * ``bias > 0`` — non-reference groups are *under-flagged*: their
      detection rate is ``base_rate × (1 − bias)``, so the Disparate
      Impact Ratio is ≈ ``1 − bias``.  Their mean ``risk_score`` is also
      shifted up by ``bias × 100`` points.
    * ``bias < 0`` — non-reference groups are *over-flagged*.

    Because ``true_outcome`` is group-independent, any disparity the
    audits report is the injected ``bias`` and nothing else — which is
    what makes this a clean verification fixture.

    Parameters
    ----------
    n : int
        Number of records.
    groups : sequence
        Group labels; ``groups[0]`` is the reference group.
    group_props : sequence, optional
        Sampling proportions for the groups (default uniform).
    n_areas : int
        Number of areas; must be ≥ the number of groups.
    base_rate : float
        Reference-group favourable-outcome rate, in ``[0, 1]``.
    bias : float
        Injected disparity, in ``[-1, 1]``.
    seed : int
        Reproducibility seed.

    Returns
    -------
    pandas.DataFrame
        Columns: ``area``, ``group``, ``true_outcome``, ``detected``,
        ``risk_score`` — directly consumable by the morie.fairness
        audits.
    """
    import pandas as pd

    groups = list(groups)
    G = len(groups)
    if G < 2:
        raise ValueError("need at least two groups")
    if not (0.0 <= base_rate <= 1.0):
        raise ValueError("base_rate must be in [0, 1]")
    if not (-1.0 <= bias <= 1.0):
        raise ValueError("bias must be in [-1, 1]")
    if n_areas < G:
        raise ValueError("n_areas must be >= the number of groups")

    rng = np.random.default_rng(seed)
    if group_props is None:
        props = np.full(G, 1.0 / G)
    else:
        props = np.asarray(group_props, dtype=float)
        if len(props) != G:
            raise ValueError("group_props must have one entry per group")
        props = props / props.sum()

    gi = rng.choice(G, size=n, p=props)
    group = np.array([groups[i] for i in gi], dtype=object)

    # Areas are segregated — each area belongs to a single group.
    area_group = np.arange(n_areas) % G
    areas_by_group = {i: np.where(area_group == i)[0] for i in range(G)}
    area_idx = np.array([rng.choice(areas_by_group[i]) for i in gi])
    area = np.array([f"area_{a:02d}" for a in area_idx], dtype=object)

    # true_outcome: group-INDEPENDENT Bernoulli(base_rate).
    true_outcome = (rng.random(n) < base_rate).astype(int)

    # detected: reference group at base_rate, others scaled by (1 - bias).
    det_rate = np.where(gi == 0, base_rate, base_rate * (1.0 - bias))
    det_rate = np.clip(det_rate, 0.0, 1.0)
    detected = (rng.random(n) < det_rate).astype(int)

    # risk_score: SSL-style 0-500; non-reference groups shifted by bias.
    loc = np.where(gi == 0, 250.0, 250.0 + bias * 100.0)
    risk_score = np.clip(rng.normal(loc, 40.0), 0.0, 500.0)

    return pd.DataFrame({
        "area": area,
        "group": group,
        "true_outcome": true_outcome,
        "detected": detected,
        "risk_score": risk_score,
    })
