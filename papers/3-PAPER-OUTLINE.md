# MORIE — 3-paper JSS-format split

Anchor template: Bach et al. (2024), *J. Stat. Softw.* v108i03,
"DoubleML: An Object-Oriented Implementation of Double Machine
Learning in R" (and its Python twin).

LaTeX class: `jss.cls` v3.4 (Zeileis) — installed locally via
R-framework texmf. Compile with `pdflatex` + `bibtex`.

Sole author: Vansh Singh Ruhela (ORCID 0009-0004-1750-3592),
Centre for Criminology and Sociolegal Studies, University of
Toronto.

Hawkes paper: **stays as a separate 4th methodology paper**
(not part of this 3-paper split). It's a method-specific paper
(self-exciting point processes), not a software-package paper.

---

## Paper 1 — Theoretical (the MRM framework)

**Title:** "The MRM Framework: A Multi-Source Statistical
Foundation for Canadian Carceral, Police, and Oversight Data"

**Directory:** `papers/mrm-formulations-paper/` (existing,
restructured to JSS class)

**Sections:**

1. **Introduction.** Carceral data landscape in Canada;
   fragmentation across SIU / OTIS / TPS / CCRSO; the need
   for a unified statistical spine. Cites Zorro Medina
   (2023), Doob & Sprott, Webster & Doob, etc. ~3 pages.

2. **The five Canadian sources.** OTIS A01-RCDD; SIU IAP; Ontario
   SIU full corpus; Toronto Police Service open data; CCRSO.
   Schemas, units, temporal coverage, access. ~5 pages.

3. **Identification strategy.**
   - 3.1 Staggered two-way-fixed-effects estimator (Callaway
         & Sant'Anna 2021, Goodman-Bacon 2021, Sun & Abraham
         2021, Athey & Imbens 2022).
   - 3.2 Leads-and-lags Granger diagnostics for the parallel-
         trends assumption.
   - 3.3 Identification under heterogeneous treatment effects.
   - ~6 pages with formal mathematical statements.

4. **Mechanism categorisation.**
   - 4.1 Deterrence (Beccaria, Becker, Nagin).
   - 4.2 Routine activities (Cohen & Felson).
   - 4.3 Certainty (Apel, Loughran).
   - Tie each mechanism to specific estimands in §3.
   - ~3 pages.

5. **The inequality-effects-of-criminal-law framing.**
   Following Zorro Medina (2023, §2.3). Maps the framework
   onto Canadian racial / Indigenous / socioeconomic
   inequality patterns. ~3 pages.

6. **Application: OTIS Mandela findings (2026).** 12.5% →
   20.6% provincial-segregation Mandela-torture rate
   2023→2025; comparison with Sprott-Doob's federal SIU
   9.9% baseline. ~4 pages with tables and figures.

7. **Implementation in morie.** Brief — *cites Paper 2 (R) and
   Paper 3 (Python)* — points the reader at the software
   companion papers. ~1 page.

8. **Conclusion.** ~1 page.

**Estimated length:** ~26 pages.

**Status:** Existing `main.tex` is 1553 lines (~25 pages
typeset). Restructuring to JSS class, tightening philosophical
digressions, formalising the mathematical statements in §3,
and adding the bibliography. ~3-4h of work.

---

## Paper 2 — morie for R

**Title:** "morie: Multi-domain Open Research and Inferential
Estimation in R"

**Directory:** `papers/morie-r-paper/` (rename from existing
`morie-paper/`)

**Sections** (mirrors v108i03):

1. **Introduction.** Position morie alongside DoubleML (Bach
   et al. 2024), MatchIt (Ho et al. 2011), survey (Lumley
   2004), hdm (Chernozhukov et al. 2016), grf (Tibshirani
   et al. 2023), AIPW (Zhong & Naimi 2021). What morie
   provides that the ecosystem doesn't — namely, the MRM
   modules as a curated sociolegal application and a
   coherent dual-language API. ~3 pages.

2. **Getting started.**
   - 2.1 Installation (CRAN + r-universe).
   - 2.2 Motivating example (OTIS A01-RCDD): negative-
         binomial mixed model + IRM via morie wrapping
         DoubleML. ~3 pages.

3. **Module taxonomy.**
   - 3.1 Causal inference (`estimate_att`, `estimate_atc`,
         `estimate_cate`, `estimate_gate`, `estimate_aipw`,
         `estimate_irm` wrapping `DoubleML::DoubleMLIRM`).
   - 3.2 Survey sampling and weights.
   - 3.3 Signal processing (butter*, sgolay_smooth, hurst).
   - 3.4 Spatial statistics.
   - 3.5 Psychometrics (alpha, omega, IRT).
   - ~6 pages.

4. **The MRM modules.** Sociolegal-specific functions:
   `mrm_otis_load`, `mrm_siu_scrape`, `mrm_tps_load`. ~3 pages.

5. **Wrapper architecture for DoubleML.** How morie's
   `estimate_irm()` composes mlr3 learners (`lrn("regr.lm")`,
   `lrn("classif.log_reg")`) with `DoubleML::DoubleMLIRM`.
   Why we wrap rather than re-implement. ~3 pages.

6. **Reproducible examples.** Three real-data examples
   (OTIS, TPS, CCS PUMF) with R> code chunks. ~5 pages.

7. **Simulation study.** Coverage / power simulations under
   varying nuisance estimation strategies (lm, glmnet,
   ranger). Mirrors v108i03 Appendix A in spirit. ~4 pages.

8. **Conclusion.** ~1 page.

**Estimated length:** ~28 pages.

**Status:** Existing `morie-paper/main.tex` is 627 lines.
Substantial new content needed for §3 (full module taxonomy),
§5 (wrapper architecture), §7 (simulation study). ~5-6h.

---

## Paper 3 — morie for Python

**Title:** "morie: Multi-domain Open Research and Inferential
Estimation in Python"

**Directory:** `papers/morie-py-paper/` (new)

**Sections:** structurally parallel to Paper 2 but Python-idiomatic.

1. **Introduction.** Position alongside DoubleML-py (Bach et
   al. 2022), causalml (Chen et al.), EconML (Battocchi et
   al. 2019), CausalPy (PyMC team), dowhy (Sharma & Kiciman).
   ~3 pages.

2. **Getting started.**
   - 2.1 Installation (pip + provenance via PEP 740).
   - 2.2 Motivating example. ~3 pages.

3. **Module taxonomy.** Python-side function inventory
   (`morie.causal.estimate_att(...)` etc.). ~6 pages.

4. **The MRM modules.** ~3 pages.

5. **Functional vs object-oriented API.** Python idioms
   (closure-based estimators, dataclasses for RichResult,
   typed signatures). ~3 pages.

6. **Reproducible examples.** Three examples. ~5 pages.

7. **Simulation study.** Parallel to Paper 2 §7. ~4 pages.

8. **Conclusion.** ~1 page.

**Estimated length:** ~28 pages.

**Status:** New paper. Reuses content patterns from Paper 2
but with Python code chunks throughout. ~6-7h.

---

## Cross-paper conventions

- **Citation style:** JSS-standard author-year (Zeileis 2023
  jss.cls handles this).
- **Code chunks:** `R>` prefix for R, `python>` for Python.
- **Tables/figures:** numbered per paper, captioned in
  sentence case.
- **Cross-paper citations:** each paper cites the other two
  by Zenodo DOI.
- **Bibitem keys:** harmonised:
  - `Ruhela2026MRM` = Paper 1 theoretical
  - `Ruhela2026MorieR` = Paper 2 R
  - `Ruhela2026MoriePy` = Paper 3 Python
  - `Ruhela2026Hawkes` = Hawkes methodology (kept separate)
  - `Ruhela2026Software` = software DOI (Zenodo 20111233)

## Total time estimate

~14–17 hours of focused writing for all three papers from
their current state. Can do them serially in this 24h block:

1. Paper 1 (Theoretical) — most existing content, 3-4h.
2. Paper 2 (R) — adds simulation, 5-6h.
3. Paper 3 (Python) — built parallel to Paper 2, 6-7h.

**Recommendation:** confirm this outline; then I execute
Paper 1 first (most independent, most reusable existing
content), surface the first PDF for your review, then
proceed.
