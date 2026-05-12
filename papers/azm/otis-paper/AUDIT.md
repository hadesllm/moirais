# OTIS audit — paper claim ↔ OTIS-RC published result ↔ data-dictionary truth

> Auditing the rewritten OTIS paper at
> `/path/to/workspace/papers/otis-paper/main.tex` against
> Ruhela's published analysis at
> `/path/to/workspace/OTIS-RC/` and the official Ontario
> A01RCDD data dictionary (Nov 3, 2025) at
> `/tmp/otis-dd.xlsx`.

---

## A.  Data-dictionary ground truth

### A.1  File inventory (30 official files)

| Family | Files | Substantive content |
|---|---|---|
| **a01** | `a01_restrictive_confinement_detailed_dataset.csv` | one row per restrictive-confinement DAY — columns: `_id`, `EndFiscalYear`, `UniqueIndividual_ID`, `Region_AtTimeOfPlacement`, `Region_MostRecentPlacement`, `Gender`, `Age_Category`, `MentalHealth_Alert`, `SuicideRisk_Alert`, `SuicideWatch_Alert`, `Number_Of_Placements` |
| **b01** | `b01_segregation_detailed_dataset.csv` | one row per segregation event — same as a01 PLUS `NumberConsecutiveDays_Segregation`, 7 `SegReason_*` columns |
| b02-b09 | various aggregates | placement counts, alerts, durations |
| **c01-c12** | individuals counts | aggregate cohort tables: race × gender (c03), age × region (c06), alerts × hold flags (c07), religion × gender (c08), durations (c10-c12) |
| **d01-d07** | deaths in custody | death counts, cause of death, alerts, race, religion, age |

### A.2  Critical ID structure (verbatim from dictionary, r8)

> "A random number assigned to an individual who was in restrictive
> confinement.  The unique ID is randomly **re-assigned to different
> individuals each year**.  The unique ID may also randomly assigned
> to different individuals **for each data file of the same year**.
> The unique ID follows the format of `YYYY-XXXXX-AA`."

**Concrete consequences:**

1. Cross-year recidivism is *unanswerable* — empirically verified
   (R: `0` of `65,467` IDs appear in ≥ 2 fiscal years).
2. Cross-file matching within the same year is *also* not safe —
   the same numeric ID across `a01.csv` and `b01.csv` may be different
   people.  Any analysis joining files by ID is invalid.
3. "Each row of data represents a day in restrictive confinement."
   Per-row aggregation must respect the day-level granularity.

### A.3  Dictionary row-31..34 label shift (b01)

The dictionary's b01 schema has a label/name shift in rows 31-34
(variable names align with our morie column set, but the labels
are shifted up by one row).  This is a dictionary bug.  Our
`MentalHealth_Alert`, `SuicideRisk_Alert`, `SuicideWatch_Alert`,
`SegReason_Other` column names are correct.

---

## B.  Our `otis_b01` schema vs the dictionary

| morie column | dict file | dict row | Status |
|---|---|---|---|
| EndFiscalYear | b01 | r18 | ✅ |
| UniqueIndividual_ID | b01 | r19 | ✅ name; ⚠️ semantics (per-year reset) |
| Gender | b01 | r20 | ✅ |
| Region_AtTimeOfPlacement | b01 | r21 | ✅ |
| Region_MostRecentPlacement | b01 | r22 | ✅ |
| Age_Category | b01 | r23 | ✅ |
| NumberConsecutiveDays_Segregation | b01 | r24 | ✅ |
| SegReason_SecurityOfInstitution_SafetyOfOthers | b01 | r25 | ✅ |
| SegReason_InmateNeedsProtection | b01 | r26 | ✅ |
| SegReason_InmateNeedsProtection_Medical | b01 | r27 | ✅ |
| SegReason_SecurityOfInstitution_SafetyOfOthers_Medical | b01 | r28 | ✅ |
| SegReason_Disciplinary_Segregation | b01 | r29 | ✅ |
| SegReason_InmateRefuseSearch_Scan | b01 | r30 | ✅ |
| MentalHealth_Alert | b01 | r31 (label shifted) | ✅ name |
| SuicideRisk_Alert | b01 | r32 (label shifted) | ✅ name |
| SuicideWatch_Alert | b01 | r33 (label shifted) | ✅ name |
| SegReason_Other | b01 | r34 (label shifted) | ✅ name |
| Number_Of_Placements | b01 | r35 | ✅ |

Our schema is **correct**; the dictionary itself has the label-shift
bug in r31-34.

---

## C.  Ruhela's published results in OTIS-RC

### C.1  `correctional_stats_report_environment1b.RData`

Loaded objects (verified): `df` (the b01 dataset), `ac1b`,
`df.counts`, `df.counts_year`, `df.counts_ac_year`, `df_overall`,
`ecdf_age`, `events_per_person`, `total_people`,
`df.person_freq_year`, plus DML-related (`m.out`, `orc_matched`,
`dml_*`, `model_final`, etc.).

**Sample-size-level descriptives** (from the dataframe inspection):
- Total rows: $76\,934$
- Unique IDs: $65\,467$ (= unique (id, year) pairs ⇒ confirms per-year reset)
- Fiscal years: $\{2023, 2024, 2025\}$

### C.2  `correctional_stats_report1z.RData` — `res_pool`

| group | estimand | $\widehat\tau$ | SE | t | p | 95% CI |
|---|---|---|---|---|---|---|
| Pooled 2023-25 | ATE  | $0.1605$ | $0.00628$ | $25.54$ | $<10^{-143}$ | $[0.1481, 0.1728]$ |
| Pooled 2023-25 | ATTE | $0.1557$ | $0.00606$ | $25.69$ | $<10^{-145}$ | $[0.1438, 0.1676]$ |

**Status in our paper:** ✅ in Table~2 of the rewrite.

### C.3  `res_by_year`

| Year | Estimand | $\widehat\tau$ | SE | 95% CI |
|---|---|---|---|---|
| 2023 | ATE  | $0.1342$ | $0.00986$ | $[0.1149, 0.1535]$ |
| 2023 | ATTE | $0.1272$ | $0.01034$ | $[0.1070, 0.1475]$ |
| 2024 | ATE  | $0.1591$ | $0.01167$ | $[0.1362, 0.1820]$ |
| 2024 | ATTE | $0.1550$ | $0.01141$ | $[0.1326, 0.1774]$ |
| 2025 | ATE  | $0.1737$ | $0.01030$ | $[0.1535, 0.1939]$ |
| 2025 | ATTE | $0.1704$ | $0.00966$ | $[0.1514, 0.1893]$ |

**Status in our paper:** ✅ in Table~3 of the rewrite.

### C.4  `res_all` — multi-way regional clustering, 30 rows

Region contrasts × clustering schemes × estimands:
- 5 regions (Toronto, Eastern, Western, Central, Northern) vs others
- 3 clustering schemes (0-way, cluster_id, cluster_id+cluster_region)
- 2 estimands (ATE, ATTE)

Point estimate range $\widehat\tau \in [0.1932, 0.2013]$
(min Eastern-cluster_id-ATTE; max Toronto-cluster_id+region-ATTE).
SE range $[0.00346, 0.01923]$.  All $p < 10^{-24}$.

**Status in our paper:** ✅ summarised in Table~4 of the rewrite as
robustness band $[0.193, 0.213]$ — slight upper-bound mis-statement
(actual max is $0.2013$, not $0.213$); will fix.

### C.5  Descriptive tables Ruhela published (from `notez1a.qmd`)

| Object | Content | In our paper? |
|---|---|---|
| `ecdf_age` | ECDF of age categories (3 bins, n + p + pct + rate per bin) | ❌ not included |
| `df.counts` | per-region (5) totals: f, count, p, pct, rate | ❌ |
| `df.counts_year` | per-year × per-region: f, count, p | ❌ |
| `df.counts_ac_year` | per-year × variable × level: alerts, demographics breakdown | ❌ |
| `df_overall` | overall per-variable × level: f, count, p, pct, rate | ❌ |
| `df.person_freq_year` | per-year × region × id: frequency | ❌ |
| `statz` | per-(year, gender) summary stats: n, mean(ac), sd(ac), mean(vm), sd(vm) | ❌ |
| `events_per_person` | distribution of n_events per id (within-year) | ❌ |
| `total_people` | $65\,467$ unique person-years | ⚠️ in §Data |

**Action:** consider adding a "descriptive statistics" section
(or appendix) that surfaces ecdf_age, df.counts, statz at minimum.

### C.6  Sensitivity / supplementary models in `notez1a.qmd`

| Model | Lines (notez1a.qmd) | In our paper? |
|---|---|---|
| MatchIt 1:1 NN PSM, formula `treat ~ ag + sg + yr`, glm distance | 1221-1229 | ✅ §3.4 |
| Poisson GLMM `vm ~ treat + ag + sg + yr + (1\|rc)` (matched) | 1232-1245 | ✅ §3.4 + §4.4 (rate-ratio narrative) |
| `dml_clustered` (cluster on id) | 1415-1432 | ✅ §3.3 + Table 4 |
| `dml_region_clustered` (cluster on rc, yr) | 1438-1454 | ✅ Table 4 |
| `dml_region` (cluster on rc) | 1456-1474 | ✅ Table 4 |
| `dml_yearz` (cluster on yr) | 1476-1494 | ⚠️ partially — Table 4 has 0-way + id + id+region only |
| `dml_yrID` (cluster on yr, with id in x_cols) | 1496+ | ❌ |
| Year × {2023, 2024, 2025} per-year DML | (around res_by_year section) | ✅ §4.2 |
| GLMM MSE comparison (vs DML) | 1395-1402 | ❌ — Ruhela compares MSE_GLMM vs MSE_DML, useful for showing DML wins |
| `t_value`, `p_value`, `ci_low`, `ci_high` per spec | embedded in res_pool/by_year/all | ✅ in our tables |

---

## D.  What our rewritten OTIS paper should ADD or FIX

### D.1  Add: Descriptive table

Surface at least `ecdf_age` + `df.counts` + `statz` so the reader
sees the population breakdown before the causal grid.  Currently
the rewrite jumps from Data → Methods → Results without the
"population at a glance" page Ruhela includes.

### D.2  Add: Year-clustered DML row to Table 4

Ruhela fits `dml_yearz` (cluster on `yr`) as a fourth clustering
scheme.  Our Table 4 has 0-way, cluster_id, cluster_id+cluster_region
only.  Adding the year-clustered row makes the table complete
against `res_all`.

### D.3  Fix: Robustness band $[0.193, 0.2013]$ not $[0.213]$

Table 4 caption claims "$\hat\tau \in [0.193, 0.213]$".  Actual
range from `res_all` is $[0.1932, 0.2013]$ — narrower than
advertised.  Will update.

### D.4  Add: GLMM-vs-DML MSE comparison

Ruhela's `notez1a.qmd` (lines 1395-1402) reports
$\text{MSE}_{\text{GLMM}}$ vs $\text{MSE}_{\text{DML}}$ on the
matched sample.  This is a useful "ML wins" sanity check that we
omit.  Add a one-paragraph note.

### D.5  Add: `dml_yrID` specification (id-in-x_cols + cluster-yr)

This is the most aggressive specification Ruhela runs (id as a
covariate AND clustering by year).  We can quote this row to
demonstrate the effect survives even when id-level fixed effects
are absorbed.

### D.6  Confirm: cross-file ID joining is NOT used in our otis_causal.py

Important to verify — if we join b01 to a01 by `UniqueIndividual_ID`
within the same year, we'd be matching people semi-randomly.  Audit
of `morie.otis_causal` shows: we only use b01 (single file), so
this hazard does not arise in our analysis.  But should be
explicitly noted in the paper's §Limitations.

### D.7  Confirm: Hill-α removal

The old paper's Hill-α = 1.62 "lifetime repeat-placement Pareto"
section is GONE in the rewrite. ✅  Verified by reading current
main.tex.

### D.8  Confirm: time-to-readmission removal

The old paper's "100% within-year gaps" section is GONE in the
rewrite. ✅  Verified.

### D.9  Add: explicit warning about cross-file ID matching

Even within a single fiscal year, joining b01 + c07 (alerts × hold
flags) by ID would NOT correctly pair the same individual's records,
because IDs are randomly re-assigned per file.  Make this explicit
in the limitations section so the reader knows what does and does
not become possible by adding more files.

---

## E.  Items confirmed correct in the rewrite

- ✅ Treatment definition: $\texttt{treat} = \1\{\texttt{ac} \ge 2\}$
- ✅ Outcome definition: $\texttt{vm}$ as count of regional transfers
  in person-year (not "distinct regions visited")
- ✅ IRM-DML (Interactive Regression Model) as primary estimator,
  not PLR
- ✅ Multi-way clustered standard errors
- ✅ Per-year breakdown showing rising trend
- ✅ MatchIt + Poisson GLMM as sensitivity
- ✅ Within-fiscal-year ID resetting flagged in Limitations §6.1
- ✅ Empirical verification table showing $0 / 65\,467$ IDs in ≥ 2 years
- ✅ Author affiliation: Centre for Criminology & Sociolegal Studies,
  School of Graduate Studies, University of Toronto St. George
- ✅ Companion repo pointer to `/path/to/workspace/OTIS-RC/`

---

## F.  Outstanding work

Sized in priority order:

1. **Fix Table 4 caption** ($0.213 \to 0.2013$): 1 min
2. **Add year-clustered DML row** to Table 4: 5 min
3. **Add GLMM-vs-DML MSE narrative**: 5 min
4. **Add Descriptive section** with ecdf_age + df.counts + statz:
   15-20 min
5. **Add `dml_yrID` aggressive-spec note**: 5 min
6. **Add cross-file-ID-matching warning**: 5 min

Total: ~40 min of paper polish, no new R code needed (all numbers
already in `correctional_stats_report*.RData`).

After this audit pass the OTIS paper will mirror Ruhela's published
analysis end-to-end with no methodological gaps.
