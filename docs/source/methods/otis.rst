Ontario Restrictive Confinement (OTIS)
======================================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE provides **250+ correctional/sociolegal functions** for Ontario
correctional system data — restrictive confinement placements, alert
statuses, regional movement patterns, recidivism, risk assessment,
sentence analytics, custody metrics, compliance monitoring, and
causal inference across fiscal years 2023-2025.

This module bridges **criminology**, **sociolegal studies**, and
**epidemiological methods** using the same DML/IPW/AIPW infrastructure
as the CPADS public health analysis. All functions are dataset-agnostic
and implemented as individual files in ``morie.fn/`` (≤7-char names).

Key function families:

- **Placement** (15): ``rpl_r``, ``rpl_a``, ``rpl_g``, ``rpl_ra``-``rpl_gt``, ``rprat``, ``rpdur``, ``rpfrq``, ``rpfst``, ``rpgap``
- **Alerts** (12): ``alrt1``-``alrt3``, ``alco``, ``altm``, ``aldur``, ``alesc``, ``altrn``, ``alprv``, ``alinc``, ``alrsk``, ``alcmx``
- **Volatility** (3): ``vol_r``, ``vol_a``, ``vol_t``
- **Recidivism** (10): ``rcdsm``, ``rcd_r``, ``rcd_a``, ``rcd_g``, ``rcdtm``, ``rcdkm``, ``rcdhz``, ``rcdcx``, ``rcdpr``, ``rcdrt``
- **Risk** (10): ``rskcl``, ``rskau``, ``rskcb``, ``rskfr``, ``rskbr``, ``rskpr``, ``rskov``, ``rskth``, ``rskdc``, ``rsktd``
- **Sentence** (10): ``sntln``, ``sntmd``, ``sntpr``, ``sntsr``, ``sntrl``, ``sntag``, ``sntgn``, ``sntrg``, ``sntdp``, ``sntvl``
- **Custody** (15): ``cstdy``, ``cstoc``, ``cstin``, ``cstsg``-``cstgp``
- **Compliance** (15): ``cmprt``-``cmpgn``, ``insprt``-``inspfr``, ``odesc``-``orank``
- **DML/Causal** (15): ``odml1``-``odml4``, ``oate1``, ``oatt1``, ``ohet1``, ``ogate``, ``ocate``, ``oipw1``, ``odid1``, ``oiv1``, ``omed1``, ``osns1``, ``oaipw``
- **Demographics** (15): ``odm_r``-``odm_y``, ``osumm``-``opair``
- **Core** (6): ``rpl``, ``astc``, ``vol``, ``rct``, ``otd``, ``oml``

Data
----

- **Source**: Ontario Ministry of the Solicitor General (data.ontario.ca)
- **Coverage**: Fiscal years 2023-2025, 5 Ontario regions
- **Records**: ~1.9M expanded placement records
- **Unit**: Individual × fiscal year × placement event

Key variables:

- ``gender`` — sex / gender (Male, Female).
- ``age_category`` — age group (18-24, 25-49, 50+).
- ``region`` — Ontario region (Central, Eastern, Northern, Toronto,
  Western).
- ``mental_health_alert`` — mental-health flag (Yes / No).
- ``suicide_risk_alert`` — suicide-risk flag (Yes / No).
- ``suicide_watch_alert`` — suicide-watch flag (Yes / No).
- ``number_of_placements`` — count of placements (integer).

Alert-State Encoding
--------------------

Three binary alerts (mental health, suicide risk, suicide watch)
produce 8 possible combinations. The codes used in the package:

- ``a1`` — mental health only (1, 0, 0).
- ``a4`` — mental health + suicide risk (1, 1, 0).
- ``a5`` — suicide risk + suicide watch (0, 1, 1).
- ``a7`` — all three alerts (1, 1, 1).
- ``a8`` — no alerts (0, 0, 0).

(Vector entries are ``mental_health, suicide_risk, suicide_watch`` in
that order. Codes ``a2, a3, a6`` cover the remaining permutations.)

The **complexity index** (``ac``) counts how many distinct alert
states a person experienced across their placement events. Higher
complexity indicates more variable alert status over time.

Methods
-------

- **Regional placement analysis** — Python ``rplace``, R ``get_region_by_age()``.
- **Alert-state encoding** — Python ``astcmb``, R ``dt_unbiased`` block.
- **Regional volatility** — Python ``volat``, R volatility section.
- **Trends over time** — Python ``rctrnd``, R temporal analysis.
- **Descriptive statistics** — Python ``otdesc``, R sections 1-4.
- **DML IRM (ATE / ATT)** — Python ``otdml``, R ``run_dml_analysis()``.
- **Propensity score matching** — Python ``morie.matching``, R ``MatchIt``.
- **AIPW (doubly robust)** — Python ``morie.causal``, R ``WeightIt`` + ``lm_robust``.
- **Mixed effects (GLMM)** — R only (``lme4``, ``glmmTMB``).
- **DHARMa diagnostics** — R only (``DHARMa``).

Usage
-----

.. code-block:: python

   from morie.otis import rplace, astcmb, otdml
   import pandas as pd

   # Load expanded placement data (via R bridge or direct)
   # df = pd.read_csv("data/cache/dt_expanded.csv")

   # Regional placement by year
   result = rplace(df, year=2024, sex="Male")
   print(result.props)

   # Alert-state complexity
   alerts = astcmb(df)
   print(alerts.summary)

   # DML causal analysis
   dml = otdml(df, outcome="Y", treatment="D")
   print(f"ATE: {dml.ate:.3f} (p={dml.ate_pval:.4f})")

References
----------

.. [SOLGEN2025] Ontario Ministry of the Solicitor General (2025).
   Restrictive Confinement Detailed Dataset. *data.ontario.ca*.

.. [Jahn2020] Jahn v. Ontario (2020). Settlement Agreement —
   Inmate Data Disclosure.

.. [Chernozhukov2018] Chernozhukov, V., Chetverikov, D., Demirer, M.,
   Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased
   machine learning for treatment and structural parameters.
   *Econometrics Journal*, 21(1), C1-C68.
