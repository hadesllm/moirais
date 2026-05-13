"""morie.mrm_primitives — reusable identification primitives for the
Multilevel Reconciliation Methodology (MRM) framework.

These primitives are the building blocks that morie modules (and any
external code) compose to construct identification strategies.  Each
primitive is a thin, well-typed function that takes a pandas
DataFrame plus a few configuration kwargs and returns a structured
result.

Five primitives currently shipped (v0.6.1):

  - :func:`gentrification_panel`        — categorical baseline-conditional
                                          gentrification coding
  - :func:`spatial_spillover_decomposition`
                                          — Spatial Durbin / SAR direct-
                                            indirect-total decomposition
  - :func:`synthetic_area_exposure`     — SAE-based exposure offset
                                          (generalises far beyond SMI)
  - :func:`threshold_specific_ordinal`  — Bayesian cumulative-logit with
                                          threshold-varying coefficients
  - :func:`score_net_residual`          — two-stage score-then-audit;
                                          residual race effect = bias

The primitives correspond to the patterns extracted from Laniyonu (2018),
Laniyonu & Goff (2021), and O'Connell & Laniyonu (2025); see ROADMAP.md
v0.6.1 + papers/mrm-formulations-paper for the academic context.
"""

from __future__ import annotations

from .gentrification import gentrification_panel
from .ordinal import threshold_specific_ordinal
from .score_net_residual import score_net_residual
from .spatial_spillover import spatial_spillover_decomposition
from .synthetic_exposure import synthetic_area_exposure

__all__ = [
    "gentrification_panel",
    "spatial_spillover_decomposition",
    "synthetic_area_exposure",
    "threshold_specific_ordinal",
    "score_net_residual",
]
