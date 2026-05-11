# SIU IAP — Federal Structured Intervention Unit Implementation Advisory Panel

*Part of {doc}`index` — MORIE's statistical-methods reference.*

The Ontario Tracking Information System (OTIS) provides **provincial**
restrictive-confinement data analysed in `morie.otis_*` modules. The
**federal** counterpart is Canada's Structured Intervention Unit (SIU)
system, which replaced administrative segregation in federal
penitentiaries in 2019.

From April 2021 to December 31, 2024, the **Structured Intervention Unit
Implementation Advisory Panel (SIU IAP)** monitored federal SIU
implementation. The panel was an oversight body, not a data-producing
body — its outputs are qualitative reports.

## Panel composition (relevant to morie)

- **Howard Sapers** — Chair. Former Correctional Investigator of
  Canada.
- **Prof. Emeritus Anthony N. Doob** — Member. Centre for Criminology
  & Sociolegal Studies, University of Toronto.
- **Prof. Jane B. Sprott** — Member. Department of Criminology,
  Toronto Metropolitan University.

Prof. Doob's membership is the proximate reason why morie analyses
of OTIS aggregate data are framed under the **Doob chi-square** name
— see `morie.otis_all_analyze.analyze_c_doob_chi2` and
`analyze_d_doob_chi2`.

## Two distinct panels

There were actually **two** Implementation Advisory Panels — the
distinction matters for citation:

- **Original SIU-IAP** (2019 → mid-2020) — chaired by Anthony N. Doob;
  exposed in morie as `siuiap.ORIGINAL_PANEL_2019_2020`.
- **Re-launched SIU IAP** (2021-04 → 2024-12) — chaired by Howard
  Sapers; exposed in morie as `siuiap.PANEL_MEMBERS` and
  `REPORTS`.

The original Doob-chaired panel "died a natural death after the
1-year terms of its members expired in mid-2020" (Sprott & Doob,
Feb 2021, footnote 7). The Sapers-chaired panel was a fresh
appointment in April 2021.

## Reports indexed

`morie.siuiap` indexes three categories of documents:

- `REPORTS` — the **5 Sapers-chaired SIU IAP panel reports** (2021-2024).
- `CRIMSL_REPORTS` — the **4 Sprott / Doob ± Iftene independent academic
  reports** (CRIMSL UToronto + Schulich Law Dalhousie, 2020-2021):
    - October 2020: *Understanding the Operation of CSC's SIUs*
    - November 2020: *Is there Clear Evidence COVID-19 Was the Cause...*
    - February 2021: *Solitary Confinement, Torture, and Canada's SIUs*
    - May 2021: *Do Independent External Decision Makers Ensure...*
- `AFFIDAVITS` — Doob's 2020 Federal Court affidavit (T-539-20).

The four CRIMSL reports + the Doob affidavit are the source data for
[`morie.sprott_doob`](sprott_doob.md) and [`morie.doob_trends`](doob_trends.md).

## Programmatic access

```python
from morie.siuiap import cite, panel_summary, REPORTS, CRIMSL_REPORTS

# SIU IAP panel reports
print(cite("final_2024"))
# SIU IAP (2024). SIU IAP Final Report. Public Safety Canada.

# CRIMSL Sprott-Doob research reports
print(cite("sprott_doob_torture_solitary_2021"))
# Jane B. Sprott, Anthony N. Doob (2021). Solitary Confinement,
# Torture, and Canada's Structured Intervention Units. Centre for
# Criminology & Sociolegal Studies, University of Toronto.

# Doob T-539-20 Federal Court affidavit
print(cite("doob_t_539_20_2020"))
# Anthony N. Doob (2020). Affidavit of Anthony N. Doob — Federal
# Court of Canada, T-539-20. Federal Court of Canada.
```

## See also

- [`sprott_doob`](sprott_doob.md) — replication of the four CRIMSL/
  Schulich SIU papers' main tables (Tables 13, 19, 22, 23 + IEDM
  analyses).
- [`doob_trends`](doob_trends.md) — replication of the Doob
  T-539-20 affidavit's CCRSO 2018 Tables 1-3 and decoupling test.
