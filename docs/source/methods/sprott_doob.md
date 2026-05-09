# Sprott-Doob CRIMSL + Schulich Law SIU analyses

`moirais.sprott_doob` replicates the analytical contribution of four
research reports authored by **Prof. Jane B. Sprott** (Toronto
Metropolitan University, formerly Ryerson) and **Prof. Anthony N.
Doob** (University of Toronto), with **Prof. Adelina Iftene**
(Dalhousie Schulich School of Law) co-authoring the May 2021 paper
on Independent External Decision Makers.

The four reports together form the most comprehensive independent
academic analysis of Canada's federal SIU system through 2021.

## The four reports

| Date | Title | Authors | Source |
|---|---|---|---|
| 2020-10 | *Understanding the Operation of CSC's SIUs* | Sprott & Doob | CRIMSL UToronto |
| 2020-11 | *Is there Clear Evidence COVID-19 Was the Cause of Problems with the SIUs?* | Sprott & Doob | CRIMSL UToronto |
| 2021-02-23 | *Solitary Confinement, Torture, and Canada's SIUs* | Sprott & Doob | CRIMSL UToronto |
| 2021-05-09 | *Do Independent External Decision Makers Ensure...?* | Sprott, Doob & Iftene | Schulich Law Dalhousie |

## Headline replicated findings

**February 2021 paper (Mandela classifier on N=1,960 SIU stays):**

| Category | Mandela Rule | % | N |
|---|---|---|---|
| Solitary Confinement | Rule 44 (≤2 hrs out, ≤15 days) | 28.4% | 556 |
| Torture | Rules 43+44 (same conditions, ≥16 days) | 9.9% | 195 |
| All other | — | 61.7% | 1,209 |

Pacific region's torture rate (39.1 per 1000 prisoners) is **22.6×**
Ontario's (1.73 per 1000).

**May 2021 paper (N=265 IEDM-reviewed stays, 380 reviews):**

- 8.7% of IEDM decisions ⇒ "remove from SIU"
- 58.9% ⇒ "remain in SIU"
- 30.3% ⇒ CSC moved the prisoner BEFORE the IEDM ruled
- 12 IEDMs varied **37.5%–85.7%** in their "should remain" rate
  (χ²=26.12, df=11, p<.01)
- **105 long-stays (76+ days) had NO IEDM record** — apparent
  compliance failure with CCRA s.37.8.
- Indigenous: 40.4% of reviewed stays. Black: 15.8% (~4× the Black
  share of the Canadian adult population). Black prisoners are
  also significantly more likely to stay 121+ days (12.7% vs 6.1%
  for whites; χ²=41.63, df=15, p<.001).

## Mandela classifier

The classifier operationalizes UN Mandela Rules 43 and 44:

```python
from moirais.sprott_doob import classify_mandela

classify_mandela(days_in_siu=20, hrs_out_of_cell_avg=1.5,
                  missed_full_4hrs_pct_of_days=100)
# {'category': 'Torture', 'rule': 'Mandela Rules 43+44', 'reason': ...}

classify_mandela(days_in_siu=8, hrs_out_of_cell_avg=1.5,
                  missed_full_4hrs_pct_of_days=100)
# {'category': 'Solitary Confinement', 'rule': 'Mandela Rule 44', ...}
```

## χ² verification

Every published χ² value is recomputed from the transcribed cell
counts. All five tested χ² statistics (Tables 11, 15, 22 from Feb
2021 plus Tables 5, 10 from May 2021) reproduce to within 0.01:

```python
from moirais.sprott_doob import verify_published_chi_squares
r = verify_published_chi_squares()
# 5/5 pass
```

Use `verify_chi2(observed_table)` to recompute χ² + p-value for any
2D contingency table.

## Mandela-RF — applying the classifier to OTIS provincial data

The Mandela 15-day threshold also applies to Ontario provincial OTIS
data via three analyzers in `moirais.otis_all_analyze`:

```python
from moirais.otis_all_analyze import (
    analyze_b05_mandela_classification,           # per-placement
    analyze_c11_mandela_classification,           # per-individual
    analyze_otis_mandela_provincial_vs_federal,   # cross-comparison
)
```

The provincial OTIS data is duration-only (no hours-out-of-cell), so
the Ontario classification is a duration-only proxy for the Mandela
threshold. Despite that limitation, the headline cross-comparison
finding is striking: the Ontario provincial **Segregation** torture
rate (per-individual, c11 view) is 12.5% in 2023, 16.5% in 2024, and
20.6% in 2025 — **over 2× the federal SIU rate of 9.9%** that
Sprott-Doob found in 2019-2020 federal data, and trending upward.

Caveats apply (different units — federal person-stays vs provincial
individuals; different Mandela operationalization — federal full
hours-out-of-cell + duration vs provincial duration-only). Use the
cross-comparison analyzer for an interpretation-aware view.

## Position in the Ruhela formulations stack

The Sprott-Doob analyses sit at the **federal national-aggregate**
level, complementing:

- **Provincial individual-level** evidence on Ontario OTIS data via
  the Ruhela formulations battery (a01/b01/b02 per-row analyzers).
- **Provincial aggregate** evidence via the b03-b09, c01-c12,
  d02-d05 RF analyzers + Doob χ² family on c-series + d-series.

`analyze_ruhela_master(include_per_row=False)` surfaces key Sprott-
Doob and Sprott-Doob-Iftene rows in §4 of the paper-ready master
report.

## Citations

```python
from moirais.siuiap import cite

cite("sprott_doob_torture_solitary_2021")
# Jane B. Sprott, Anthony N. Doob (2021). Solitary Confinement,
# Torture, and Canada's Structured Intervention Units. Centre for
# Criminology & Sociolegal Studies, University of Toronto.

cite("sprott_doob_iftene_external_decision_makers_2021")
# Jane B. Sprott, Anthony N. Doob, Adelina Iftene (2021). Do
# Independent External Decision Makers Ensure that "An Inmate's
# Confinement in a Structured Intervention Unit Is to End as Soon
# as Possible"? [Corrections and Conditional Release Act, Section
# 33]. Schulich School of Law, Dalhousie University ...
```


The two private-Drive Sprott-Doob papers (Oct 2020 + Nov 2020) have
filename author orders that differ between the two — "DoobSprott..."
and "Sprott&Doob..." respectively. Authorship attribution in
`siuiap.CRIMSL_REPORTS` follows the published front-matter on the
public CRIMSL/Schulich uploads (Sprott first); when in doubt, follow
the front matter on the actual document.
