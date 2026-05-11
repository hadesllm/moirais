Dataset-Agnostic Analysis
==========================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE can profile and analyse *any* tabular dataset without prior knowledge
of its schema via the :mod:`morie.dataset` module. This capability is essential
when working with novel administrative health data, new survey waves, or
datasets outside of CPADS.

Levels of Measurement
----------------------

MORIE follows Stevens (1946) four-level typology when classifying columns:

- **Nominal** — categories, no order. Operations: =, ≠. Examples: sex, province, ethnicity.
- **Ordinal** — ordered categories. Operations: =, ≠, <, >. Examples: Likert scale (1--5), severity grade.
- **Interval** — equal intervals, no true zero. Operations: +, −, mean. Examples: year, temperature (°C), index score.
- **Ratio** — equal intervals + true zero. Operations: +, −, ×, ÷, geometric mean. Examples: income, age, count, weight.

Inference rules used by :func:`morie.dataset.infer_measurement_level`:

1. Object / category dtype + ≤ ``ordinal_threshold`` unique values → **Ordinal**
2. Object / category dtype + > ``ordinal_threshold`` unique values → **Nominal**
3. Integer / float + exactly 2 unique values → **Nominal** (binary)
4. Integer / float + ≤ 20 unique values + name suggests rank/scale → **Ordinal**
5. Float + name contains *year*, *index*, *score* → **Interval**
6. Float or integer + min ≥ 0 → **Ratio**
7. All other numeric → **Interval**

Dataset Profiling
------------------

:func:`morie.dataset.profile_dataset` builds a :class:`morie.dataset.DatasetProfile`
containing a :class:`morie.dataset.ColumnProfile` for every column.

Each ``ColumnProfile`` records:

- ``level`` — inferred measurement level (NOIR)
- ``missing_pct`` — fraction of missing values
- ``is_binary`` — True when only two distinct non-null values exist
- ``suggested_role`` — one of ``treatment``, ``outcome``, ``covariate``,
  ``weight``, ``stratum``, ``cluster``, ``id``
- ``summary_stats`` — mean/SD (numeric) or top-k category counts (categorical)

Role detection heuristics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Treatment column candidates: binary column with a name containing
``treat``, ``cannabis``, ``drug``, ``alcohol``, ``intervention``,
``exposed``, or ``assigned``.

Outcome column candidates: column named with a suffix / prefix matching
``outcome``, ``result``, ``y_``, ``response``, ``_freq``, ``_harm``,
``_drink``, or ``disorder``.

Weight column candidates: column name containing ``weight``, ``wt``,
``pw``, or ``survey_wt``.

If the user supplies hints via ``hint_treatment``, ``hint_outcome``, or
``hint_weights``, these override the heuristic detection.

Analysis Plan Suggestion
------------------------

:func:`morie.dataset.suggest_analysis_plan` inspects the
:class:`DatasetProfile` and returns an ordered list of suggested analyses.
Each suggestion is a dict::

    {
        "analysis": "ipw_ate",
        "rationale": "Binary treatment + continuous outcome detected",
        "required_vars": {"treatment": "cannabis_use", "outcome": "drink_freq"},
        "optional_vars": {"weights": "weight_var"},
    }

The plan covers associational statistics (prevalence, χ², correlation),
causal estimates (IPW, AIPW, DML), and regression models, ordered from
simplest to most assumption-intensive.

Usage Example
--------------

.. code-block:: python

   import pandas as pd
   from morie.dataset import load_dataset, profile_dataset, suggest_analysis_plan

   df = load_dataset("data/my_survey.csv")
   profile = profile_dataset(df, hint_treatment="cannabis_use")

   # Print rich-formatted summary table
   profile.summary_table()

   # Get suggested analysis plan
   for step in suggest_analysis_plan(profile):
       print(step["analysis"], "—", step["rationale"])

CLI usage::

   morie profile-dataset --csv data/my_survey.csv

References
----------

- Stevens SS (1946). On the theory of scales of measurement.
  *Science*, 103(2684):677–680. https://doi.org/10.1126/science.103.2684.677
- Wickham H (2014). Tidy Data. *JOSS*, 59(10):1–23.
  https://doi.org/10.18637/jss.v059.i10
