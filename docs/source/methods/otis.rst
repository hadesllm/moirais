Ontario Restrictive Confinement (OTIS)
======================================

MOIRAIS provides **250+ correctional/sociolegal functions** for Ontario
correctional system data — restrictive confinement placements, alert
statuses, regional movement patterns, recidivism, risk assessment,
sentence analytics, custody metrics, compliance monitoring, and
causal inference across fiscal years 2023-2025.

This module bridges **criminology**, **sociolegal studies**, and
**epidemiological methods** using the same DML/IPW/AIPW infrastructure
as the CPADS public health analysis. All functions are dataset-agnostic
and implemented as individual files in ``moirais.fn/`` (≤7-char names).

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

.. list-table:: Key Variables
   :header-rows: 1
   :widths: 20 30 30

   * - Variable
     - Description
     - Values
   * - gender
     - Sex/gender
     - Male, Female
   * - age_category
     - Age group
     - 18-24, 25-49, 50+
   * - region
     - Ontario region
     - Central, Eastern, Northern, Toronto, Western
   * - mental_health_alert
     - Mental health flag
     - Yes/No
   * - suicide_risk_alert
     - Suicide risk flag
     - Yes/No
   * - suicide_watch_alert
     - Suicide watch flag
     - Yes/No
   * - number_of_placements
     - Count of placements
     - Integer

Alert-State Encoding
--------------------

Three binary alerts produce 8 possible combinations:

.. list-table::
   :header-rows: 1
   :widths: 10 15 15 15 20

   * - Code
     - Mental Health
     - Suicide Risk
     - Suicide Watch
     - Interpretation
   * - a1
     - 1
     - 0
     - 0
     - Mental health only
   * - a4
     - 1
     - 1
     - 0
     - MH + suicide risk
   * - a5
     - 0
     - 1
     - 1
     - Suicide risk + watch
   * - a7
     - 1
     - 1
     - 1
     - All three alerts
   * - a8
     - 0
     - 0
     - 0
     - No alerts

The **complexity index** (``ac``) counts how many distinct alert states
a person experienced across their placement events. Higher complexity
indicates more variable alert status over time.

Methods
-------

.. csv-table::
   :header: "Method", "Python function", "R equivalent"
   :widths: 30, 25, 25

   "Regional placement analysis", "``rplace``", "``get_region_by_age()``"
   "Alert-state encoding", "``astcmb``", "``dt_unbiased`` block"
   "Regional volatility", "``volat``", "volatility section"
   "Trends over time", "``rctrnd``", "temporal analysis"
   "Descriptive statistics", "``otdesc``", "sections 1-4"
   "DML IRM (ATE/ATT)", "``otdml``", "``run_dml_analysis()``"
   "Propensity score matching", "``moirais.matching``", "MatchIt"
   "AIPW (doubly robust)", "``moirais.causal``", "WeightIt + lm_robust"
   "Mixed effects (GLMM)", "R only", "lme4, glmmTMB"
   "DHARMa diagnostics", "R only", "DHARMa"

Usage
-----

.. code-block:: python

   from moirais.otis import rplace, astcmb, otdml
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
