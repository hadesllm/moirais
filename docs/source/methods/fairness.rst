Fairness & Disparity Audit
==========================

The ``morie.fairness`` subsystem (introduced in v0.8.0) audits
risk-assessment, recidivism, and predictive-policing systems for
racial — and other protected-attribute — disparities.

.. note::

   morie does **not** build or deploy predictive-policing systems.
   This subsystem *measures* whether an existing system encodes
   disparate treatment, so that researchers, oversight bodies, and the
   public can hold those systems accountable.

Provenance
----------

Every method here is a clean-room reimplementation, written from
*published descriptions* — no third-party code was copied:

* IBM **AI Fairness 360** metric definitions.
* The SciencesPo **Predictive-policing-Chicago** project (Lachérade,
  Szabo, Krikava & Aeby, 2021) — the Strategic Subjects List audit.
* **Barman & Barman**, *Unmasking Algorithmic Bias in Predictive
  Policing* (arXiv:2603.18987) — the temporal multi-city metrics.
* The **COMPAS** audit in pbiecek's *XAI Stories*.

Disparity metrics
-----------------

Six classical group-fairness measures, each returning a rich,
paragraph-level result (Python and R, full parity):

* ``fairness_disparate_impact`` — the EEOC four-fifths rule.
* ``fairness_demographic_parity`` — the favourable-rate gap.
* ``fairness_equalized_odds`` — true/false-positive-rate gaps.
* ``fairness_average_odds_difference`` — the mean TPR/FPR gap.
* ``fairness_gini`` — concentration of a score distribution.
* ``fairness_bias_amplification`` — the composite
  :math:`\text{BAS} = \Delta_{\text{parity}} \times G`.

Predictive-policing calibration audit
-------------------------------------

``predpol_calibration_audit`` ranks areas by the risk an algorithm
*predicts* and by their *realised* outcome rate, then tests whether the
disagreement tracks the areas' demographic composition — the signature
of disparate over-policing.  ``predpol_score_disparity`` is the
descriptive first step; the ``CityProfile`` layer makes the audit
city-agnostic (Chicago, New York, Toronto, or any registered city).

Multi-city temporal audit
-------------------------

``predpol_temporal_audit`` computes the four disparity metrics for each
``(city, period)`` cell and reports temporal instability and
cross-city divergence — bias metrics are not stable across deployment
cycles and must be re-audited per period.

Simulation framework
--------------------

The optional ``morie[sim]`` extra adds a JAX-based simulation layer:
the Noisy-OR patrol-detection model, a synthetic biased-crime-data
generator, a spatial GAN, and a CTGAN-style conditional tabular
debiaser.  JAX (Apache-2.0, CPU-first) is used in place of PyTorch to
keep the install lean.

Explainability (XAI)
--------------------

Model-agnostic explainers — ``xai_permutation_importance`` (which flags
protected features a model leans on), ``xai_partial_dependence``,
``xai_ale``, ``xai_ceteris_paribus``, and ``xai_shap_values`` — surface
*which features* drive a model's disparate predictions.
