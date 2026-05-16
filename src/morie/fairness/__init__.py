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

from .metrics import (
    fairness_average_odds_difference,
    fairness_bias_amplification,
    fairness_demographic_parity,
    fairness_disparate_impact,
    fairness_equalized_odds,
    fairness_gini,
)

__all__ = [
    "fairness_disparate_impact",
    "fairness_demographic_parity",
    "fairness_equalized_odds",
    "fairness_average_odds_difference",
    "fairness_gini",
    "fairness_bias_amplification",
]
