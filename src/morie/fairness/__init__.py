# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie.fairness — group-disparity audit toolkit.

A subsystem for *auditing* risk-assessment, recidivism, and
predictive-policing systems for racial (and other protected-attribute)
disparities.  morie does not deploy such systems — it measures whether
an existing one encodes disparate treatment, so that researchers,
oversight bodies, and the public can hold those systems accountable.

Phase A ships the disparity-metrics core
(:mod:`morie.fairness.metrics`): classical group-fairness measures —
disparate impact, demographic parity, equalised odds, average odds
difference, and the Gini coefficient — each returning a rich,
paragraph-level :class:`~morie.fn._richresult.RichResult`.

Later phases add the generalised predictive-policing audit module, the
multi-city temporal analysis, the JAX simulation framework, and the
explainability (XAI) layer.
"""
from __future__ import annotations

from .cityprofile import (
    CityProfile,
    apply_profile,
    get_city,
    list_cities,
    register_city,
)
from .metrics import (
    fairness_average_odds_difference,
    fairness_bias_amplification,
    fairness_demographic_parity,
    fairness_disparate_impact,
    fairness_equalized_odds,
    fairness_gini,
)
from .predpol import (
    predpol_aggregate_areas,
    predpol_calibration_audit,
    predpol_score_disparity,
)
from .simulation import noisy_or_detection, simulate_biased_crime_data
from .temporal import predpol_temporal_audit

__all__ = [
    # disparity metrics
    "fairness_disparate_impact",
    "fairness_demographic_parity",
    "fairness_equalized_odds",
    "fairness_average_odds_difference",
    "fairness_gini",
    "fairness_bias_amplification",
    # predictive-policing calibration audit
    "predpol_calibration_audit",
    "predpol_aggregate_areas",
    "predpol_score_disparity",
    "predpol_temporal_audit",
    # simulation primitives
    "noisy_or_detection",
    "simulate_biased_crime_data",
    # city-agnostic data layer
    "CityProfile",
    "register_city",
    "get_city",
    "list_cities",
    "apply_profile",
]
