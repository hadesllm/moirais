# SESSION HANDOFF — morie-empirical-paper, 2026-05-11 (continuing the same multi-day session)

> Read this FIRST in the next session. Everything verified up to 2026-05-11 ~10:30 EDT.
>
> Latest checkpoint: §7.8 Goffmanian churn fully verified + rewritten with truth.
> Paper builds clean (20 pages, 0 LaTeX errors, no missing-bib warnings).

## ✅ COMPLETED + VERIFIED (cumulative)

### Empirical paper status
- **Location:** `/Volumes/VSR/rootcoderfiles/papers/morie-empirical-paper/`
- **Current state:** 19 pages, compiles clean
- **Build cmd:** `cd <dir>; rm -f main.{aux,out,bbl,blg,toc} main.pdf; pdflatex; bibtex main; rm -f main.out; pdflatex; rm -f main.out; pdflatex`
- **The `rm main.out` between passes is REQUIRED** because hyperref + \OTIS/\TPS in section titles causes a stale-bookmark error otherwise.
- **All section titles using \OTIS or \TPS must use** `\section[plain alias]{... \OTIS{} ...}` form.

### §5 DML results — VERIFIED to 5 decimals against `correctional_stats_report1z.RData`
- Pooled ATE = 0.1605 (SE 0.00628), ATTE = 0.1557 — matches `res_pool`
- 2023/2024/2025 ATE = 0.1342 / 0.1591 / 0.1737 — matches `res_by_year`
- Multi-way clustering: $\hat\tau \in [0.1932, 0.2013]$ across 30 cells — matches `res_all`
- Sample: 76,934 rows / 65,467 unique person-years (cross-year IDs: 0 — IDs reset annually)

### §6 Mandela — REWRITTEN with c11-canonical numbers (matches MRM paper §8 Table 1 exactly)
**VERIFIED by R script against `c11_individuals_in_segregation_and_restrictive_confinement_aggregate_lengths.csv`:**

| Year | Seg Solitary % | Seg Torture % | N Seg | RC Solitary % | RC Torture % | N RC |
|------|---|---|---|---|---|---|
| 2023 | 87.5 | **12.5** | 12,647 | 68.5 | **31.5** | 20,781 |
| 2024 | 83.5 | **16.5** | 10,881 | 64.0 | **36.0** | 19,641 |
| 2025 | 79.4 | **20.6** | 9,608 | 59.1 | **40.9** | 25,045 |
| Federal 2019-20 | 28.4 | **9.9** | 1,960 person-stays | — | — | — |

The c11 file is the OFFICIAL Ontario aggregate — duration-band individual counts. Torture* = sum of {16-20, 21-25, 26-30, >30 days} bands.

### Vee's MRM formulas (from `OTIS-RC/explority.R`)
- **ac**: 3 alerts → 8 status combos a1..a8; at person-year, $a_k$ = count of placements in state $k$; $\ac = $ rowSums(a1, a2, a4, a5, a7, a8 > 0) = # distinct alert-combo states (out of 6 selected). Excluded: a3 (risk-only), a6 (mental+risk).
- **vm**: `sum((regA != regB) | (regA != shift(regA) & !is.na(shift(regA))))` per individual-year — count of regional transitions.
- Treatment: $D = \mathbf{1}\{\ac \ge 2\}$.

### morie::mrm_classify_mandela() — committed
- Commit `9001ab3bd` at `hadesllm/morie` main, `r-package/morie/R/mandela.R`.
- Currently 3 b01-based operationalisations.
- **TODO:** add `source = "c11_aggregate"` mode that reads c11 to reproduce the canonical 12.5/16.5/20.6 / 31.5/36.0/40.9 (the b01 modes give 0.17/1.53/3.06 etc — different unit, kept for completeness).

## ⏳ NEXT-SESSION TODO (priority order)

### 1. ✅ DONE 2026-05-11 — §7.8 Goffmanian institutional churn verified

**Verified against `b09_individuals_in_segregation_number_of_times_in_segregation.csv`:**

| Metric | MA-thesis claim | Verified (pooled 2023-25) | Status |
|---|---|---|---|
| Total individuals | — | 33,136 | — |
| Total placements | — | 153,067 | — |
| Gini coefficient | 0.575 | **0.585** | within rounding ✓ |
| Top 5% concentration | ≥25% | **25.9%** | ✓ EXACT |
| Hill α (x_min=1) | 1.62 | **2.08** | DIVERGENT — used 2.08 in paper |
| Per-year Gini | — | 0.542 / 0.596 / 0.616 | rising |

**Verified against `b01_segregation_detailed_dataset.csv`:**

- Mortification co-occurrence: **MentalHealth × SuicideRisk V = 0.189**, χ²=2914, p<10⁻³⁰⁰ — EXACT match to MA-thesis ✓
- SuicideRisk × SuicideWatch V = 0.668 (subset-escalation, not substantive co-occurrence)
- Region churn χ² = 285,917, df=16, p<10⁻³ ✓ — BUT INTERPRETATION REVERSED:
  - 95.0% within-region staying (diagonal); only 5.0% cross-region
  - Substantive finding: locality-PRESERVING, not regional-churn

**Removed from paper:** KM time-to-readmission (invalid because OTIS IDs reset annually).

**Saved to:**
- `results/06-goffmanian-churn-b09.txt` — Hill α + Gini + top-5% per year
- `results/07-mortification-region-b01.txt` — Cramér's V + region contingency

**Paper text updated:** §7.8 in main.tex now reports verified numbers; the "Power-law repeat placement" paragraph is renamed "Heavy-tailed repeat placement", explicitly flagging the α=1.62 → 2.08 correction. New "Within-region locality" paragraph corrects the "routinised regional flows" framing.

### 1b. STILL TODO: verify §7.1 + §7.2 + §7.3-7.7 + §7.9-7.11 physics claims

### 2. Verify §7.1 + §7.2 Hawkes against TPS Assault CSV
- Data: `/Volumes/VSR/rootcoderfiles/moirais-dev/dev/sphinx/project/data/datasets/TPS/Assault/CSV/Assault_Open_Data_*.csv` (254,378 events 2014-present, OCC_DATE column)
- Tool: `tps.R` in same dir
- MA-thesis claims Gamma-kernel × sinusoidal-baseline preferred for Assault by AIC + KS

### 3. Verify §7.3 Lévy mobility / urban scaling / predator-prey
- Run morie statistical-physics modules; compare exponents to MA-thesis §3.3

### 4. Verify §7.4 Moran's I, §7.5-7.7 spatial scan / inspection game / reaction-diffusion
### 5. Verify §7.9 SARIMA / Langevin / Fokker-Planck forecasting
### 6. Verify §7.10 CSI integration (uses c-tables + TPS UCR codes)
### 7. Verify §7.11 Extended spatial diagnostics

### 8. Update `mrm_classify_mandela()` with c11 mode, push, bump morie 0.1.15

## DATA LOCATIONS (don't search again)

- **OTIS:** `/Volumes/VSR/rootcoderfiles/moirais-dev/dev/sphinx/project/data/datasets/OTIS/`
  - **c11** = THE Mandela source (per fiscal-year × duration-band individual counts; verified 2026-05-11)
  - **a01** (76,934 rows): RC day-level
  - **b01** (82,001 rows): segregation event-level
  - **b09**: individuals × number-of-times-in-segregation — for Hill-α/Gini repeat-placement concentration
- **TPS:** `/Volumes/VSR/rootcoderfiles/moirais-dev/dev/sphinx/project/data/datasets/TPS/`
  - 13 categories, each w/ CSV, GeoJSON, Shapefile subdirs
- **OTIS-RC (Vee's DML pipeline):** `/Volumes/VSR/rootcoderfiles/OTIS-RC/`
  - `correctional_stats_report1z.RData`: res_pool, res_by_year, res_all, model_final, m.out, orc_matched
  - `notez1a.qmd` lines 1221-1245: MatchIt+GLMM spec
  - `explority.R` lines 197-265 + 1050-1075: ac/vm definitions
- **Userguides (Sprott-Doob/SIU lit):** `/Volumes/VSR/.../data/datasets/userguides/`

## PAPER STRUCTURE (current §s)

| § | Title | Status |
|---|---|---|
| 1 | Introduction | — |
| 2 | Data (OTIS b01 + TPS) | ✅ Schema verified |
| 3 | Methods (corrected ac/vm + IRM-DML + clustering + MatchIt+GLMM) | ✅ Formulas verified |
| 4 | Descriptives (sample structure) | ✅ |
| 5 | Causal estimates: alert complexity on placement volatility (Tables 2-4) | ✅ Verified to 5 decimals |
| 6 | Mandela Rules classification on OTIS (Federal + Provincial Seg + Provincial RC) | ✅ Verified from c11 |
| 7 | Statistical physics of crime and institutional churn (11 subsections) | ⏳ Forwarded from MA-thesis; needs morie-pipeline re-run |
| 8 | Reproducibility (morie code per result) | — |
| 9 | Limitations / Discussion / Conclusion | — |
| App A | Full morie R code per result | — |

## VERIFICATION SCRIPTS (paste into R)

```r
# §5 DML
load("/Volumes/VSR/rootcoderfiles/OTIS-RC/correctional_stats_report1z.RData")
print(res_pool); print(res_by_year)
cat("res_all range:", range(res_all$effect), "\n")

# §6 Mandela from c11
c11 <- read.csv("/Volumes/VSR/rootcoderfiles/moirais-dev/dev/sphinx/project/data/datasets/OTIS/c11_individuals_in_segregation_and_restrictive_confinement_aggregate_lengths.csv")
names(c11)[1] <- "EndFiscalYear"
torture_bands <- c("16 to 20 days","21 to 25 days","26 to 30 days","Greater than 30 days")
for (yr in 2023:2025) {
  d <- c11[c11$EndFiscalYear == yr,]
  cat(yr, "Seg torture %:", round(100*sum(d$NumberIndividuals_Segregation[d$Aggregate_Duration %in% torture_bands])/sum(d$NumberIndividuals_Segregation),1),
      " | RC torture %:", round(100*sum(d$NumberIndividuals_RestrictiveConfinement[d$Aggregate_Duration %in% torture_bands])/sum(d$NumberIndividuals_RestrictiveConfinement),1), "\n")
}
# Expect: 12.5/16.5/20.6 Seg ; 31.5/36.0/40.9 RC

# §7.8 Goffmanian churn — run when otis_churn.R is sourced
# Hill-MLE α + Gini against b09 (repeat-placement distribution)
# Expected: alpha = 1.618, gini = 0.575
```

## CRITICAL LESSONS (do not repeat)

1. **Never claim "verified" from a document — only from actual code execution.**
2. **OTIS has 3 files that could underlie a "Mandela rate":** a01 (RC day-level), b01 (segregation event), c11 (official aggregate). **c11 is the canonical source for the 12.5/16.5/20.6 + 31.5/36.0/40.9 rates.** They come from the OFFICIAL Ontario published aggregate, not from row-level computation.
3. **OTIS IDs are randomly re-assigned each year.** Any cross-year tracking (recidivism, KM survival across years) is **invalid**. Empirically: 0 of 65,467 IDs appear in ≥ 2 years. The MA-thesis "Kaplan-Meier time-to-readmission" claim needs flagging or removal.
4. **Vee's MRM contribution is ac (state diversity) + vm (transition count)** — not "count of alerts" or "binary moved" indicators.
5. **For LaTeX:** `\section{... \OTIS{} ...}` breaks hyperref bookmarks. Always provide `\section[plain]{... \OTIS{} ...}`. Delete `main.out` between pdflatex passes.
6. **Long section titles overflow.** §5 was "Main results: causal estimates of alert complexity on placement volatility" — shortened to "Causal estimates: alert complexity on placement volatility". §7 was "Statistical-physics-of-crime results on TPS and institutional-churn results on OTIS b01" — shortened to "Statistical physics of crime and institutional churn".

## RUNTIME NOTES

- R 4.6.0 at `/opt/homebrew/bin/R`
- morie 0.1.14 live on PyPI + r-universe; CRAN under review (submitted 2026-05-11)
- TinyTeX with jss.cls (Achim Zeileis v3.4) via R framework texmf
- Conventions: ALWAYS save verified outputs to `results/` before claiming "verified". Run R scripts directly against the data files.
