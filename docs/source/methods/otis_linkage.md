# OTIS Linkage Constraints — Read Before Doing Any Individual-Level Analysis

*Part of {doc}`index` — MOIRAIS's statistical-methods reference.*

**TL;DR**: OTIS `UniqueIndividual_ID` is **NOT** a stable person identifier. It is randomly re-assigned per fiscal year **and** per dataset file. Any analysis claiming individual-level linkage across years, or across datasets within the same year, is artifactual.

## What the official dictionary says (verified 2026-05-08)

From the Ontario MCSCS data dictionary v2.0 (resource `d83fe893-9634-4794-a0c1-c17bf619a95a`, last modified 2025-11-19):

> `UniqueIndividual_ID`: A random number assigned to an individual (format: **YYYY-XXXXX-AA**), where **YYYY reflects the year at the end of the fiscal year reporting period** [calendar year for d-series], `XXXXX` is a sequence, and `AA` is a dataset acronym (`RC`, `SG`, `DC`).
>
> **The unique ID is randomly re-assigned to different individuals each year. The unique ID may also be randomly assigned to different individuals for each data file of the same year.**

`_id` (the CKAN row ID) "**cannot be used to link datasets, as the same _id will likely represent different records across different datasets**."

## What this means in practice

- ✅ **"Person X had N placements in fiscal year Y"** — valid (within
  one dataset, one fiscal year).
- 🔴 **"Same person was in segregation in 2023 and again in 2024"** —
  not measurable: IDs don't carry over.
- 🔴 **"Same person appears in both a01 (RC) and b01 (Segregation)
  within FY 2024"** — not measurable: IDs are also re-randomized
  between files.
- ✅ **"% of person-years involving multiple regions"** — valid
  (intra-year Goffmanian mobility).
- ✅ **"Aggregate placements per region per year"** — valid (no
  individual linkage required).
- ✅ **"Distribution of placements per individual within a fiscal
  year"** — valid.

## Empirical confirmation

Run `.venv-314/bin/python -c "from moirais.otis_datasets import load_otis_dataset; ..."`:

- **a01** (76,934 rows, 65,467 unique IDs): 0 IDs span >1 fiscal year. YYYY-prefix == EndFiscalYear in 100% of rows.
- **b01** (82,001 rows, 33,136 unique IDs): 0 IDs span >1 fiscal year.
- **a01 ∩ b01 (any FY)**: 0 shared `XXXXX-AA` suffixes — the AA differs (`RC` vs `SG`), and even stripping AA the suffix space doesn't overlap.
- The "same suffix" workaround fails: e.g. `2023-00002-RC` and `2024-00002-RC` show different demographic profiles (age 50+ → 25-49 → 50+ — biologically impossible for one person). This is the documented re-randomization at work.

## Which analyses are valid

### Within-year individual-level (intra-FY Goffmanian)
- `moirais.otis_churn.within_year_placement_count(b01)` — distribution of placements per (id × FY) cell. (50.3% of person-years have multiple placements; Gini = 0.432.)
- `moirais.otis_churn.within_year_region_diversity(b01)` — distinct regions per (id × FY) cell. (3.8% of person-years span multiple regions.)
- Alert co-occurrence within an FY (chi² + Cramer's V): mortification cluster.
- Disciplinary × medical-protective overlap within an FY.
- Region × alert state-richness within an FY.

### Aggregate / population-level (no individual linkage)
- `moirais.otis_churn.repeat_placement_concentration(b09)` — population-wide placements-per-individual distribution from binned aggregate data; Goffmanian heavy tail via Gini + power-law.
- `moirais.otis_churn.embedding_distribution(b02)` — total-days-in-segregation distribution; lognormal vs Pareto AIC.
- Year-over-year aggregate trend tests (rate ratios, Pettitt change-point).
- Demographic contingency tables (race × region, gender × age, etc.) within and across years (using counts, not linkage).

### Causal estimands (intra-year)
- `moirais.otis_causal.otis_irm_dml` — IRM-DML with `cluster_cols=["yr"]` (treats year as the cluster, not individuals).
- `moirais.otis_causal.otis_aipw`, `otis_psm`, `otis_ipw` — all operate on the (treat, vm) pair within FY.

## Which analyses are INVALID — never trust their output

If you see code that does any of these, treat the result as artifact:

- `groupby("UniqueIndividual_ID")["EndFiscalYear"].diff()` — every group has exactly one year, so `.diff()` returns NaN/0; "100% same fiscal year" is the artifact, not Goffman.
- Any "time-to-readmission across fiscal years" claim from OTIS data.
- Any "same person also showed up in dataset X" join across `a01` and `b01` (or any pair).
- Stripping the YYYY prefix to use `XXXXX-AA` as a cross-year key — empirically broken (see above), and explicitly disclaimed by the dictionary.

## How we got here (2026-05-08)

A prior `moirais.otis_churn.time_to_readmission()` reported "100% of gaps are within the SAME fiscal year — Goffman's cyclical inmate dynamic." That was the artifact, not the dynamic. The function has been renamed `within_year_placement_count` with corrected semantics. `cross_region_churn` was renamed `within_year_region_diversity` with corrected interpretation (the math was always intra-year; only the framing was wrong).

## Sources

- [Ontario Data Catalogue — Data on Inmates in Ontario](https://data.ontario.ca/dataset/data-on-inmates-in-ontario)
- [Data Dictionary XLSX v2.0 (2025-11-03)](https://data.ontario.ca/dataset/09f7fc65-d3bb-4ca8-8b84-1cdc3ef73c36/resource/d83fe893-9634-4794-a0c1-c17bf619a95a/download/od-restrictiveconfinement-segregation-deaths-dd20251103.xlsx)
- Local copy: `data/datasets/OTIS/OTIS_DATA_DICTIONARY.md`
