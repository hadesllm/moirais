"""morie.laniyonu — policing & corrections methodology from Laniyonu et al.

Three reproducible-replication modules, each calling out to the
v0.6.1 :mod:`morie.mrm_primitives` building blocks:

  - :func:`gentrification_policing`   — Laniyonu (2018) UAR 54(5):898–930
  - :func:`smi_force_disparity`       — Laniyonu & Goff (2021) BMC Psych
                                          21(1):500
  - :func:`actuarial_risk_disparity`  — O'Connell & Laniyonu (2025)
                                          Race & Justice 15(3):428–453

Each module composes the corresponding MRM primitives and adds the
paper-specific wrapper (column-name conventions, reporting structure,
sensitivity grid).  Users who already have data in canonical form can
call the primitives directly; this module is the "I have the data
the way the paper has it" entry point.
"""

from __future__ import annotations

from .gentrification_policing import gentrification_policing

__all__ = [
    "gentrification_policing",
    # smi_force_disparity and actuarial_risk_disparity ship in the
    # follow-up v0.6.1 patches (tasks #236 and #237).
]
