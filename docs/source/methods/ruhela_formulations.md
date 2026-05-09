# Ruhela Formulations

A **Ruhela formulation** is a (treatment, outcome, covariates) design
choice for a specific OTIS dataset, paired with the full causal-method
battery applied to it. The framework was designed end-to-end by
**Vansh Singh Ruhela** (hadesllm); see the attribution block at
the head of `moirais.otis_causal` for the full lineage.

## Vocabulary

| Term | Meaning |
|---|---|
| **RF** | Ruhela formulation: one (T, Y, covariates) design choice for a dataset |
| **RDF** | Ruhela Dual Formulation: an RF paired with a Naive-arm sensitivity contrast |
| **DLRM** | Doob-Levinsky-Ruhela-Medina (methodology attribution acronym, narrow) |
| **Ruhela method battery** | The 10-estimator suite applied to any RF |

## Method battery (10 estimators)

For every RF that resolves to per-row panel data with a binary T and
numeric Y, the battery runs:

1. **IPW (Hájek)** — single-robust on propensity, Lunceford-Davidian sandwich SE
2. **AIPW (RRZ doubly-robust)** — cross-fitted IF plug-in
3. **g-computation (Robins 1986)** — single-robust on outcome model + bootstrap SE
4. **PSM 1:1 NN (Austin 2011)** — nearest-neighbour with caliper, returns ATT
5. **PSM subclass (Rosenbaum-Rubin 1983)** — 5-strata weighted ATE
6. **IRM-DML (Chernozhukov 2018)** — cross-fitted RF nuisance, reports ATE + ATTE + ATC, cluster-robust SE
7. **PSM → IRM-DML (match_first)** — the author's MatchIt-then-DoubleML pipeline
8. **ATC AIPW** — E[Y(1)−Y(0) | D=0] via cross-fitted IF reweighted on the D=0 stratum
9. **PLR DML (Chernozhukov 2018)** — partially linear regression, FWL-residualised, homogeneous-effect θ
10. **SuperLearner-stacked AIPW** — convex stack of RF + ridge + GLM + mean (xgboost optional), NNLS weights

Plus **multi-SE comparison** on the IRM-DML primary estimate: pooled
(iid), cluster on EndFiscalYear, cluster on UniqueIndividual_ID,
multi-way (year × id) — same point estimate, four standard errors.

Plus **propensity calibration** (Platt or isotonic) on the propensity
scores in IPW / AIPW / SuperLearner-AIPW. Reports Brier-score
diagnostic.

## Per-row formulations (panel data)

For a01 and b01 the canonical RF is `T_high_ac → Y_vm_count` paired
with a Naive arm (`any-flag → vm-binary`).

```python
from moirais.otis_all_analyze import (
    analyze_a01_ruhela_formulations,
    analyze_b01_ruhela_formulations,
    analyze_b02_ruhela_formulations,
    analyze_a01_ruhela_alt_gender,
    analyze_a01_ruhela_alt_age,
    analyze_a01_ruhela_alt_toronto,
    analyze_a01_ruhela_per_year,
    analyze_b01_ruhela_per_year,
)
```

## Aggregate formulations (count outcomes)

For aggregate datasets the analog is Poisson + NB GLM with IRR.

```python
from moirais.otis_all_analyze import (
    analyze_b03_ruhela_aggregate,
    analyze_b06_ruhela_aggregate,
    analyze_b07_ruhela_aggregate,
    analyze_c01_ruhela_aggregate,
    analyze_c03_ruhela_aggregate,
    analyze_c04_ruhela_aggregate,
    analyze_c06_ruhela_aggregate,
    analyze_c07_ruhela_aggregate,
    analyze_c09_ruhela_aggregate,
)
```

## Doob chi-square companion

```python
from moirais.otis_all_analyze import analyze_c_doob_chi2, analyze_d_doob_chi2
```

## Constraints

- OTIS IDs are year-locked; no cross-year individual linkage by
  design. See `methods/otis_linkage.md`.
- Aggregate IRRs are confounded with cell-size; per-row a01/b01
  formulations are the canonical individual-level contrast.
- Federal counterpart: SIU IAP (Doob, Sapers, Sprott et al.). See
  `methods/siuiap.md`.
