# morie v0.3.0 tracker

Items deferred from v0.2.1 to v0.3.0.  Populated as the Wave 1 + Wave 2
agent batch returns; each agent's SKIPPED rows are merged here.

## Scope of v0.2.1 batch

14 textbook formula suites, **275 callables**, Python + R parity targeted:

| # | Suite | Callables | Agent ID prefix | Wave | Status |
| - | ----- | --------: | --------------- | ---- | ------ |
| 1 | Schabenberger spatial-stats   | 20 | aaf4c96b | 1 | **DONE 20/20** (1 partial: sglm Gaussian-only) |
| 2 | Kosorok empirical-process     | 20 | afe15f13 | 1 | **DONE 20/20 (verified)** |
| 3 | ML foundations                 | 20 | a4f685ab | 1 | **DONE 20/20** (Py smoke; R code-review; R named `ml_<short>.R` — deviates from convention) |
| 4 | Ghosal Bayesian-nonparametric | 20 | aa5d5dcd | 1 | **DONE 20/20** (Py smoke; R parity in ONE file `ghosal_bnp.R`) |
| 5 | Horowitz econometrics         | 20 | a4a9579c | 1 | **DONE 20/20** (Py smoke; R parity in ONE file `horowitz.R`) |
| 6 | Montesinos genomic-ML         | 20 | af436b81 | 2 | **DONE 20/20** (Py smoke; R code-review) |
| 7 | Armstrong **spatial-voting** (not robust-stats) | 20 | a6687cf7 | 2 | **DONE 20/20** (Py smoke; R parse-only) |
| 8 | Deep learning                 | 20 | aff588a0 | 2 | **DONE 20/20** (Py smoke OK; R code-review only) |
| 9 | "Missing-data stats" (MISNAMED — actually QMC/copulas/EVT) | 20 | a254226d | 2 | **DONE 20/20** — Py smoke-ran |
| 10 | LLM architecture              | 20 | a00095e3 | 2 | **DONE 20/20** (Py + R written; sandbox blocked execution) |
| 11 | Fauzi (15)                    | 15 | aeb7196e | 2 | **DONE 15/15** — see notes |
| 12 | Gibbons-remaining             | 20 | af9df92c | 2 | **DONE 20/20** (Py smoke-tested; R code-review only) |
| 13 | Rangayyan biomedical-signal   | 20 | a0d62279 | 2 | **DONE 20/20** (Py smoke-ran; R parse-only) |
| 14 | Time-series advanced          | 20 | abf4f4d4 | 2 | **DONE 20/20** (Py byte-compile; R code-review only; R-side in ONE file not per-callable) |

## v0.3.0 SKIPPED items (to populate per agent return)

Each agent was instructed: if a formula is intractable in this session
(e.g. needs CmdStan, full MCMC infrastructure, or specialty C library not
available), set the Python body to
`raise NotImplementedError("<name>: needs <missing-dep>; tracker for v0.3.0")`
and mark SKIPPED in its summary.  Those rows land here.

### Schabenberger (Schabenberger & Gotway — spatial statistics)

**20/20 landed Py + R, but one callable is partially implemented.**

**v0.3.0 follow-up:**

1. **`sglm` — non-Gaussian families** (HIGH PRIORITY).  Current impl
   handles **Gaussian-identity only**.  Binomial/Poisson/Gamma links
   raise `NotImplementedError("sglm: family=... needs PQL/Laplace;
   tracker for v0.3.0")`.  v0.3.0: implement PQL (penalised
   quasi-likelihood) or Laplace approximation per Schabenberger Ch
   5.4, or wrap `spaMM` / `mgcv::gam(... s(coords, bs='gp'))` on R
   and a Py equivalent.
2. **`cokrg` — multivariate cokriging beyond bivariate** — current is
   co-located simple cokriging for 2 variables.  v0.3.0: extend to
   ≥3 variables, heterotopic (disjoint) sampling, and full linear
   model of coregionalization.
3. **`nstat` — full deformation model** — current uses Sampson-Guttorp
   moving-window kernel.  v0.3.0: implement the full deformation
   approach (iterative non-metric MDS for non-stationary covariance).
4. **`sarla` MLE vs 2SLS** — current uses concentrated-likelihood ML
   (Anselin 1988).  v0.3.0: add a `method="2SLS"` option for the
   `spatialreg::stsls` equivalent (will disagree by O(1/n)).
5. **R smoke-test runtime verification** — sandbox blocked Rscript;
   agent did formula-identity comparison only.  Re-run all 20 in
   v0.3.0 with `devtools::load_all()`.
6. **Package-level `import morie.fn` blocked by missing statsmodels** —
   agent bypassed via stand-alone harness.  Post-wave unify step
   (#161) must ensure statsmodels is in `Suggests`/`requires` so
   imports work in lean envs.

### Kosorok (Kosorok 2008 — empirical processes & semiparametric inference)

**All 20 callables landed Py + R, smoke-test verified.**  Agreement
exact to floating-point for 18/20; two exceptions noted below.

**v0.3.0 enrichments queued from this suite:**

1. **ksr03 KS p-value algorithmic gap** — scipy's `kstest` uses
   Marsaglia–Tsang–Wang asymptotic; R's `ks.test` uses exact CDF
   for n ≤ 100.  Statistic agrees exactly; p-value differs (~0.305
   vs 0.349 on canonical fixture).  Either route both languages to
   the asymptotic form, or document the algorithmic difference in
   the docstring with a note that v0.3.0 will provide an
   `exact_pvalue=True` option.
2. **ksr07 / ksr08 bootstrap RNG cross-language** — Py uses
   `numpy.random.default_rng(seed)`; R uses `set.seed`; they cannot
   share an RNG stream.  Current SEs agree within ~2e-2 at B=2000
   (within Monte-Carlo tolerance).  v0.3.0: add a deterministic
   pseudo-bootstrap mode that uses a shared SHA-keyed sequence so
   Py/R bootstrap draws are bit-identical for verification builds.
3. **ksr10 Huber-M** — currently direct IRWLS (k=1.345); could
   adopt `MASS::rlm` / `statsmodels.robust.RLM` for richer
   ψ-functions (Tukey, Hampel).  Optional enrichment.
4. **ksr19 Cox PH** — current implementation handles single-covariate
   with Breslow ties via Newton-Raphson.  v0.3.0: extend to
   multi-covariate with Efron tie-handling (use `survival::coxph` /
   `lifelines.CoxPHFitter`).
5. **ksr15 one-step DR** — currently location-IF (Kosorok Ch 7
   example).  v0.3.0: plug in `econml` for richer DR scenarios.

### ML foundations (Mohri-Rostamizadeh-Talwalkar + Hastie-Tibshirani-Friedman)

**All 20 callables landed.**  Python live smoke-tested via
`/tmp/morie-feature/_smoke_ml.py`.  R parity by code review only.
testthat infrastructure also added at
`r-package/morie/tests/testthat/test-ml-foundations.R`.

**v0.3.0 follow-up:**

1. **R file-naming deviation** — agent wrote R files as
   `r-package/morie/R/ml_<short>.R` (e.g., `ml_linrg.R`) instead of
   the per-callable `<short>.R` convention used by every other suite.
   This will break the post-wave unify step (#161) if not handled.
   v0.3.0 (or before commit): rename to bare `<short>.R` for
   consistency, OR explicitly document the namespaced naming and
   adjust the unify scripts.
2. **`xgbst` backend fallback** — Py falls back to `sklearn.ensemble.HistGradientBoosting*` when `xgboost` unavailable; output
   includes `backend="sklearn_histgb"` field.  R prefers xgboost then
   falls back to gbm.  v0.3.0: pin both sides to xgboost (CRAN +
   PyPI both have it) for deterministic cross-language agreement, or
   formally declare the backend-divergence in the API.
3. **Deterministic CI parity callables** — `linrg`, `pcadm`,
   `confm`, `rocau` are bit-portable to 1e-10.  v0.3.0 should add
   these specifically to a cross-language parity-CI step.  The other
   16 are not bit-portable (RNG / solver / convention differences).
4. **`rgztn` lambda convention** — sklearn vs glmnet scale lambda
   differently (glmnet by 1/n).  v0.3.0: document both conventions
   in the docstring with a conversion formula.
5. **`polrg` feature-order** — sklearn `PolynomialFeatures` uses
   lexicographic; R orders pure powers then crosses.  Coefficient
   vectors won't align by position.  v0.3.0: normalise to a single
   canonical order on both sides.
6. **Stochastic-callable list (no bit-portability)**: `rfens`,
   `gbens`, `xgbst`, `tsnrd`, `rndsr`.  Same deterministic-mode fix
   queued for Kosorok/DL/Armstrong/Montesinos applies here.
7. **R smoke-test runtime verification** — Rscript blocked.  Run in
   v0.3.0.

### Ghosal (Ghosal & van der Vaart 2017 — Bayesian nonparametrics)

**All 20 callables landed.**  Python live smoke-tested.  R parity in
ONE file `r-package/morie/R/ghosal_bnp.R` (same single-file pattern as
horowitz / time-series / llm-arch).

**Notable pre-existing handling:** `ghsrv` had a "Survival spatial
health" stub.  Agent kept back-compat alias `ghsrv =
ghosal_survival_beta_process` so existing `__init__.py` import
(`from .ghsrv import ghsrv`) still resolves; the underlying function
is now the proper Hjort (1990) beta-process posterior.

**Bayesian Gibbs chain runtimes:**
- `ghdpm`: 120-iter Neal (2000) Algorithm-3 collapsed Gibbs (40 burn, 80 saved)
- `ghhbp`: 400-iter Escobar-West augmentation Gibbs
- `ghcls`: Laplace-approximation probit-GP
- `ghwav`: hand-rolled Haar DWT + BayesThresh (no pywt dep)

**v0.3.0 follow-up:**

1. **R-side file split** — `ghosal_bnp.R` houses all 20.  Refactor to
   per-callable files matching the bare convention.
2. **Production Bayesian fits** — Gibbs chains are sub-second
   (lightweight default).  v0.3.0: add `method="long"` parameter
   routing to longer chains or PyMC/Stan for publication-grade.
3. **`ghdpm`, `ghhbp`, `ghbvm`, `ghstk` MCMC cross-language** — RNG
   stream differences give ~10% Py/R disagreement.  Same
   deterministic-mode SHA-keyed seed stream fix queued across suites
   applies here.
4. **R smoke-test runtime verification** — Rscript blocked.

### Horowitz (semiparametric / nonparametric econometrics)

**All 20 callables landed.**  Python live smoke-tested: every callable
recovers its DGP target (Robinson 0.6% error, Ichimura β̂ within 8e-4
of truth, Cox β̂ within 12%, ATE 1.66 vs target 1.5, etc.).  R parity
in ONE file `r-package/morie/R/horowitz.R` (same single-file pattern
as time-series + llm-arch, NOT per-callable).

**Bugs fixed during implementation:**
1. `hrzt1` and `hrzn1` had bootstrap-within-bootstrap exponential blow-up
   → fixed with `_bootstrap=False` recursion guards.
2. `hrzd1` Cox-PH was using a Hessian with a sign error; fixed to use
   observed-information matrix (positive-definite Newton step).
3. `hrzn1` was calling `np.math.factorial` (removed in NumPy ≥2);
   switched to `math.factorial`.

**v0.3.0 follow-up:**

1. **R-side file split** — `horowitz.R` houses all 20.  Refactor to
   per-callable files (`hrzk1.R`, `hrzk2.R`, ...) matching the bare-name
   convention used by every other suite.
2. **`hrzb1` Manski cube-root SE** — non-normal asymptotics correctly
   flagged via `RichResult.warnings`.  v0.3.0: document this in the
   public API with a how-to-interpret note.
3. **`hrzn1` NPIV parametric fallback** — current uses 2SLS fallback
   for n<50.  v0.3.0: implement the full penalised-B-spline NPIV
   (Newey-Powell) instead of fallback.
4. **R smoke-test runtime verification** — Rscript blocked.  Run
   `r-package/morie/horowitz.R` canonical tests in v0.3.0 with a
   real R session.
5. **R `@keywords internal`** — agent intentionally marked internal
   (no @export).  Post-wave unify (#161) must decide whether to
   @export these for the public API or keep internal.

### Montesinos (Montesinos-Lopez et al. — *Multivariate Statistical Machine Learning for Quantitative Genetics* Springer 2022)

**All 20 callables landed Py + R.**  Python live-tested on a 10×6
marker matrix (full suite passes); R parity by code-review only
(Rscript blocked).

**Design choices logged:**

- `gmatv` (VanRaden G matrix) is the **single source of truth**;
  `gblpf`, `mtgbl`, `mrkvr` all import it.  Same architectural pattern
  as Fauzi's `_silverman_h` shared helper.
- Bayesian callables (`blasf`, `brdgf`, `bglup`) run **short fixed
  Gibbs chains** (200-300 iters / 50-100 burn-in) for sub-second
  runtime.  `warnings` field on RichResult documents this; production
  users redirected to `BGLR` for publication-grade posteriors.
- DL callables (`dlgen`, `cnnge`, `rnnge`, `trfge`) are NumPy-only /
  base-R-only (NO torch / keras dep).  `trfge` uses random fixed
  projections + ridge head rather than trainable attention.
- `svmge`/`rfgen`/`gbgen` wrap sklearn (Py) vs e1071/randomForest/gbm
  (R) → expect ~10% disagreement (different package implementations).

**v0.3.0 follow-up:**

1. **Production-grade Bayesian fits** — current short Gibbs (200-300
   iters) is for sub-second runtime, not publication.  v0.3.0: add
   `method="long"` parameter routing to `BGLR` (R) and a Py
   equivalent for production posteriors.
2. **`trfge` transformer with trainable attention** — current is
   random fixed projections.  v0.3.0: use `torch` (gated import) for
   a real attention head trained on the genomic data.
3. **Shared-helper export pattern** — `gmatv` is used by 4 other
   callables.  Post-wave unify step (#161) must ensure file load
   ordering puts `gmatv.py` / `gmatv.R` first, OR factor to
   `_helpers_montesinos`.  Same collation concern as Fauzi.
4. **Ensemble Py/R agreement** — `rfgen` and `gbgen` disagree by
   ~10% because sklearn vs randomForest/gbm have different default
   hyperparams + tree-building algorithms.  v0.3.0: pin both to
   identical hyperparams (max_depth, n_estimators, learning_rate,
   subsample, RNG seed) for sub-1% agreement.
5. **R smoke-test runtime verification** — Rscript blocked.  Run in
   v0.3.0.

### Armstrong (Armstrong-Bakker-Carroll-Hare-Poole-Rosenthal — *Analyzing Spatial Models of Choice and Judgment*, Chapman & Hall 2014/2020)

**MISIDENTIFIED in my dispatch brief:** I told the agent "Armstrong
robust-stats / errors-in-variables".  The agent correctly identified
from the JSON entries that this is the **spatial-voting / ideal-point
estimation** textbook (NOMINATE, optimal classification, MDS, IRT,
Romer-Rosenthal agenda-setter, etc.).  The `statsmodels.robust` /
`simex` hint in my prompt was wrong; agent ignored it.

**All 20 callables landed.**  Python: 16 stubs replaced + 4 pre-existing
preserved (`idlpt`, `wnom`, `cndrc`, `agset`).  `agset.py` had a
hardcoded `statistic=0.0` bug that the agent **fixed**.  Live Py smoke
on 17/17 testable callables.  R parity created from scratch.

**v0.3.0 follow-up:**

1. **Iterative-algorithm Py/R agreement** — `irtsp`, `bysid`, `unfdl`,
   `dwnmn` use language-native iterative routines (NR, MCMC,
   SMACOF, Kalman); current agreement target is ~3-5%.  v0.3.0:
   pin both sides to identical iteration tolerances + start values
   for ~1e-6 agreement.
2. **`bysid` MCMC reproducibility** — Bayesian IRT (`bysid`) draws
   posterior samples; Py uses `numpy.random.default_rng(seed)`, R
   uses base R RNG.  ~5% disagreement.  Same deterministic-mode fix
   queued for Kosorok ksr07 + Deep-learning stochastic callables
   applies here.
3. **R smoke-test runtime verification** — Rscript blocked.  Re-run
   in v0.3.0.
4. **Cross-callable `dwnmn` Kalman-RTS** — current implementation is
   1D state-space; v0.3.0: extend to multi-dimensional Kalman with
   covariance smoothing.

### Deep learning (Goodfellow/Bengio/Courville)

**All 20 callables landed.**  Python: 18 stubs replaced + 2 pre-existing
real impls preserved (`xavir` = xavier_init returning DescriptiveResult,
`diffu` = heat_diffusion PDE solver).  For `diffu`, agent added the
spec DDPM `diffusion_forward` ALONGSIDE the existing `heat_diffusion`.
R parity line-for-line; Rscript blocked by sandbox.

**v0.3.0 follow-up:**

1. **`xavir.py` return-type conflict** — pre-existing impl returns
   `DescriptiveResult`, new convention is `RichResult`.  v0.3.0:
   migrate xavir to RichResult to harmonise with the rest of `fn/`.
2. **`diffu.py` two-function module** — same pattern as the LLM-arch
   `kvcmp.py`: legacy + spec function coexist.  Decide whether to
   keep both under `diffu` (with heat_diffusion as the canonical
   export) or split into `diffu` (DDPM) + `heatd` (PDE solver).
3. **Stochastic-callable RNG parity** — `heinz`, `drpfw`, `lstmc`,
   `grucl`, `mhatf`, `trfbl` use language-native RNGs so weight
   draws differ Py vs R.  Math operators are identical; only the
   weights/dropout masks/samples differ.  v0.3.0: add a
   `deterministic=True` mode that uses a shared SHA-keyed seed
   stream for verification builds (same pattern as Kosorok
   ksr07/ksr08).
4. **R smoke-test runtime verification** — Rscript blocked.  Re-run
   in v0.3.0 with `devtools::load_all()`.

### Missing-data stats — **CRITICAL: JSON filename is misleading**

**The JSON `missing_stats_formula_index.json` does NOT contain missing-data
methods.**  It catalogues **resampling, QMC, nonparametric regression,
dependence modelling, and extreme-value statistics** (btsrp/jkest/permt
bootstrap-jackknife-permutation, impsm/antth/latnh/sobls importance &
QMC, rkhsc/pspln/tpspn nonparametric regression, copul/vines copulas,
extvm/gpfit/retlv extreme-value).

**All 20 callables landed** Py + R; Python smoke-ran cleanly; R
parity authored by translation + code review (sandbox blocked Rscript).

**v0.3.0 must-do follow-up:**

1. **Locate or write the REAL missing-data spec** — if Vee genuinely
   wanted Little-Rubin MCAR/MAR/MNAR tests, multiple imputation,
   Rubin's rules, FIML, MICE, MissMech, EM imputation, IPW for
   missingness — that's a *separate* batch of 20 callables.  Either
   discover the correct JSON in userguides or commission a new
   `missing_data_formula_index.json` and run an agent on it in
   v0.3.0.  Without this the morie surface still lacks core
   missing-data inference tooling.
2. **`mcint.py` and `cntrl.py` legacy preservation** — these already had
   real (unrelated) implementations: `mcint` = spatial Monte-Carlo
   integration; `cntrl` = string-theory central charge.  The agent
   added the JSON-spec semantics as **R-only** callables
   (`mcint_crude`, `cntrl_estimator`) with different signatures.
   v0.3.0: decide whether to rename to disambiguate (`mcint_spatial`
   + `mcint_crude`; `cntrl_string` + `cntrl_estimator`).
3. **`gpfit.py` dispatcher** — Py now dispatches between legacy
   spatial-Pareto signature and new GP-POT signature based on
   argument names.  Document the dispatcher publicly.
4. **R smoke-test runtime verification** — sandbox blocked Rscript.
   Run `devtools::load_all()` + canonical fixtures in v0.3.0.
5. **`strat.py`** — legacy Cochran stratified-survey implementation
   preserved.  No conflict, but document the dual-purpose nature.

### LLM architecture (Vaswani 2017 + Dao 2022 + Su 2021 + Zhang-Sennrich 2019)

**All 20 callables landed.**  Pure-numpy on Py, pure base-R on R
(no torch / no keras).  R parity in ONE file
`r-package/morie/R/llm_arch.R` (same pattern as time-series).

Notable: `flshA` uses the online-softmax FlashAttention recurrence
(Dao 2022), with the canonical test asserting bit-equivalence to
naive `softmax(QKᵀ/√d)V`.

**v0.3.0 follow-up:**

1. **Case-collision risk — `flshA.py`** — the short name has an
   uppercase `A`.  This is the SAME class of bug the SIU CI fix in
   v0.2.1 (commit 4ebcdce8f) cleaned up — file resolves on macOS APFS
   but on Linux CI the casing must match git index exactly.  v0.3.0:
   rename to `flsha` (all lowercase) and update all imports +
   references.  CHECK BEFORE PUSHING v0.2.1 that the LLM-arch commit
   doesn't reintroduce mixed-case fn module names.
2. **R-side file split** — `llm_arch.R` houses all 20 functions.
   Refactor to per-callable files for convention consistency.
3. **`kvcmp` two-function module** — agent kept both legacy
   `kv_cache_compress` (TurboQuant) AND new `kv_cache_management` in
   the same module.  Decide: keep both (with the legacy
   one as the canonical export) or split into two modules in v0.3.0.
4. **Smoke-test runtime verification** — pytest + Rscript blocked by
   sandbox.  Run:
   ```
   pytest tests/test_llm_arch.py
   Rscript tests/test_llm_arch_pyr_parity.R
   ```
   in v0.3.0 to confirm green.
5. **R `@export` tags absent** — agent intentionally did not export
   (left NAMESPACE untouched per instruction).  Post-wave unify step
   (#161) must `roxygen2::roxygenize()` to populate NAMESPACE with
   these.

### Fauzi (Ahmad Reza Fauzi — KernelInference / kernel-density nonparametric inference)

**All 15 callables landed Py + R:** fzkdf, fzbrd, fzqnt, fzedg, fzmrl,
fzmrb, fzksm, fzcvm, fzsgn, fzwlc, fzhok, fzmis, fzsrv, fzhdc, fzlst.

**v0.3.0 follow-up items from this suite:**

1. **Smoke-test runtime verification** — the agent's sandbox blocked
   `python` / `R` execution, so the "Py ≈ R within 1e-6" claim is
   *by-construction* (matched formulas, same Silverman bandwidth, same
   kernels) rather than measured.  Run all 15 canonical inputs through
   both sides in v0.3.0 and confirm.
2. **Collate fragility** — `fzkdf.R` defines `.morie_silverman_h`
   used by every other `fz*.R`.  R loads files alphabetically so this
   happens to work, but the safer fix is an explicit `Collate:` field
   in `DESCRIPTION` pinning the fz file order.  Or extract the helper
   to a shared `r-package/morie/R/_helpers_fauzi.R` that's `Collate`'d
   first.  Queue for v0.3.0.
3. **fzhdc subsample branch** — for n > 63 the Hájek-projection
   estimator switches to a seeded RNG subsample; Py/R will only agree
   when both honour the seed.  Add an explicit `seed` parameter at the
   public API and document.

### Gibbons-remaining (Gibbons & Chakraborti — nonparametric inference)

**All 20 callables landed Py + R.**  Python smoke-tested (all pass);
R parity verified by code inspection (sandbox blocked Rscript).

**v0.3.0 follow-up:**

1. **R smoke-test re-run** — sandbox blocked `Rscript`, so R parity is
   code-review only.  Re-run the canonical fixtures in `_gibbons_smoke.R`
   in v0.3.0 and confirm.
2. **`ordlt.py` had a pre-existing real implementation** —
   `ordered_logit` (proportional-odds MLE) was already there, not a
   stub.  Agent preserved it and added `ordered_alternatives_test`
   (Jonckheere-Terpstra) as a second exported callable.  v0.3.0:
   decide whether to keep both under `ordlt` or split into separate
   modules `ordlt` (Jonckheere-Terpstra, Gibbons-spec) +
   `ordlg` (proportional-odds, legacy).
3. **`cov2s` and `plcmt` tie-handling** — Python and R parity holds for
   tie-free inputs.  With ties, R uses `findInterval(left.open=TRUE)`;
   need to confirm Py behaves identically when ties present.

### Rangayyan (Rangayyan — Biomedical Signal Analysis 3rd ed)

**All 20 callables landed Py + R.**  Python live smoke-tested (5/5
QRS peaks recovered on canonical ECG, all 20 RichResults populated);
R was parse-only (sandbox blocked Rscript).

**Bug fix shipped during implementation:** `rgfir` needed
`len(x) > 3*order` guard for `filtfilt` padding; agent fell back to
`lfilter` (single-pass) on short signals.  Already in code.

**v0.3.0 follow-up:**

1. **`rgwav` DWT filter set divergence** — Py uses `pywt` default
   filter (db4), R uses `wavelets` default which differs.  ~5% Py/R
   disagreement.  v0.3.0: force both to identical filter by passing
   `wavelet="db4"` explicitly on both sides; document the choice.
2. **`rgqrs` peak-refinement ±1 sample** — Pan-Tompkins R-peak
   detection can differ by ±1 sample between Py and R because of
   integer-window edge effects in the differentiation step.  v0.3.0:
   add a sub-sample parabolic-interpolation peak refinement that
   eliminates the integer-rounding gap.
3. **`rgpsd` / `rgeeg` / `rgcoh` Welch averaging** — Py uses
   `scipy.signal.welch` one-sided density scaling; R uses manual Welch
   averaging on `stats::fft`.  ~1e-3 disagreement (acceptable for
   PSD estimation).  v0.3.0: factor the manual Welch into a shared
   helper used by both sides.
4. **R smoke-test runtime verification** — Rscript blocked.
   Re-run in v0.3.0 with `devtools::load_all()` + the canonical
   ECG fixture.

### Time-series advanced (Hyndman-Athanasopoulos + Tsay)

**All 20 callables landed.**  Python: 17 stubs replaced + 3 already-real
modules preserved (`garch.py`, `coitg.py`, `cohrc.py`).  R-side
implemented as a **single file** `r-package/morie/R/time_series_advanced.R`
containing all 20 functions (not the per-callable file pattern the
other suites use).

**v0.3.0 follow-up:**

1. **R-side file split** — refactor `time_series_advanced.R` into 20
   per-callable files matching the other suites' pattern.  This is a
   convention enforcement, not a math issue.
2. **Smoke-test runtime verification** — sandbox blocked `python3` /
   `Rscript`, so Py is byte-compile-verified only and R is
   code-review only.  Run:
   ```
   cd /tmp/morie-feature && pytest tests/fn/test_garch.py \
       tests/fn/test_egarch.py tests/fn/test_midas.py
   ```
   in v0.3.0 to confirm runtime success.
3. **GARCH-family ~1% MLE discrepancy** — different GARCH optimisers
   (`arch` in Py vs `rugarch` in R) give slightly different
   `omega`/`alpha`/`beta` (within 1%).  Documented via the `method`
   field on RichResult.  v0.3.0 enrichment: pin both to identical
   start-values + identical optimiser tolerance for bit-identical
   agreement when needed.
4. **ucmod classical decomposition** — current path matches
   `stats::decompose` exactly.  v0.3.0: add the STL-decomposition
   variant (`statsmodels.tsa.seasonal.STL` / `stats::stl`) as a
   parameter choice.
5. **regms Markov-switching** — Hamilton EM init randomness gives
   ~1% cross-language difference.  v0.3.0: provide a deterministic
   k-means-on-residuals initialiser for reproducibility.

## v0.3.0 implementation priorities

In addition to the SKIPPED rows above, v0.3.0 already has these
items queued from `papers/PAPERS_v0.2.0_WORK_SUMMARY.txt` PART B:

1. Provincial Rule-43 ∩ Rule-44 **c11 individual-cumulative** spectrum
   (cross-references b01 alert flags within fiscal year via the stable
   `YYYY-XXXXX-SG` identifier).  Function:
   `mrm_otis_mandela_individual_cumulative_spectrum()`.
2. **Federal duration-only sub-rate** back-calculation from Sprott-Doob
   2023 Table 2 (would close the federal `A` column in the
   provincial-vs-federal Mandela spectrum table cleanly; partial
   bounds added in v0.2.1).
3. **CCRSO fetcher / parser** for the federal correctional data layout
   beyond SIU (McNamara primitives generalisation).
4. Multi-category JIT Hawkes re-fits (currently only Assault landed
   under the v0.2.1 JIT path; AutoTheft / BreakAndEnter / Homicide /
   Robbery / Shooting / TheftFromMV / TheftOver still on the legacy
   n=1,500 sub-sampled estimates).
5. C/Rcpp port of the Hawkes O(n²) inner loops (currently Numba-JIT
   parallel @prange; would help for n ≳ 50,000).
6. Sum-of-exponentials Hawkes kernel (Markovian-embedding of fat-tailed
   excitation, retains O(n) recursion with M accumulators).

## Caveats on "passes" in v0.2.1

Even where an agent reports DONE for a callable, the verification
target is a smoke-test of Py ≈ R numerical agreement on a small
canonical input.  Independent peer-review against the source textbook
formulas is queued for v0.3.0 in batches of 50 callables at a time.

## Post-Wave unify outcome (task #161)

- **Python**: `morie.fn/__init__.py` regenerated programmatically from each
  module's validated `__all__` (identifier regex + Python-keyword filter).
  Final: 36,402 import lines; 68,760 public symbols on `morie.fn`.
  53 modules had broken `__all__` content (e.g. `atfla` literally had a
  Lao Tzu quote string as `__all__`) — those fell back to top-level
  `def` names.  43 modules had no extractable exports and were skipped.
  All 275 textbook-suite callables import cleanly.
- **R**: NAMESPACE regenerated via direct regex sweep of `#' @export`
  tags (skipped `roxygen2::roxygenize` because its `load_source` step
  failed on Rtsne unavailable).  374 functions now exported.  All 237
  R files parse cleanly.  Single-file packs (horowitz.R, llm_arch.R,
  time_series_advanced.R, ghosal_bnp.R) are not split — logged for
  v0.3.0.
- **R `@importFrom` optional-package issues** logged for v0.3.0:
  `caret`, `dbscan`, `e1071`, `glmnet`, `pROC`, `randomForest`, `rpart`,
  `Rtsne` should all be in `Suggests:` not `Imports:`, with runtime
  `requireNamespace()` guards in the function bodies (most agents did
  this already).

## Cross-references

- `docs/designexptr_coverage.md` (already populated for the mrm
  pedagogical surface)
- `papers/PAPERS_v0.2.0_WORK_SUMMARY.txt` (the v0.1.x → v0.2.0 ledger)
- Each formula-index JSON under
  `/Volumes/VSR/rootcoderfiles/data/datasets/userguides/other/`

## v0.4.0-alpha deployment plan

These items are queued for the v0.4.0-alpha release pass (after the
v0.3.0 release lands on CRAN + PyPI + r-universe):

1. **arXiv preprint deposit for all 5 papers** — bundle TeX source +
   compiled PDF + abstract for each paper; submit to stat.ME (mrm,
   empirical, hawkes), cs.MS (morie-r-paper, morie-py-paper).
   First-time submitter endorsement may be needed per category.
   After acceptance, add `eprint = "26XX.XXXXX",
   archivePrefix = "arXiv"` fields to each paper's `Ruhela2026*`
   bib self-cites alongside the existing Zenodo `doi = "..."`
   entries.  Bump to v0.4.0-alpha after eprints land so the bib
   entries reflect the new ID.

2. **Per-component licensing audit on every dependency** — the
   v0.3.0 license split (Python `MIT OR Apache-2.0`; R `GPL-2.0-only`)
   needs a one-pass verification that no Python dependency is
   GPL-3-only (which would force Python under GPL).  Update
   `LICENSING_ANALYSIS.md` (currently historical) into a forward-
   looking compliance document for the dual-license era.

3. **Linux DEB + RPM packaging pipeline** — beyond the v0.3.0
   `install.sh`, add a GitHub-Actions workflow that builds and
   publishes DEB (Debian/Ubuntu) and RPM (Fedora/RHEL) packages on
   tag.  Sign with a project GPG key; host on Cloudsmith or
   self-host.

4. **Deterministic-mode SHA-keyed seed stream** — cross-suite fix
   for the stochastic-callable Py↔R bit-portability gap (Kosorok
   bootstrap, Deep-learning dropout/LSTM, Armstrong Bayesian IRT,
   Montesinos Bayesian/DL, Ghosal MCMC).

5. **Locate or write the true missing-data spec** — replaces the
   v0.2.1 `missing_stats_formula_index.json` (misnamed; actually
   QMC/copulas/EVT).

## v0.4.0-alpha scope (locked by Vee 2026-05-12)

Five concrete deliverables for the alpha milestone:

1. **arXiv preprints for all 5 papers** (mrm, morie-r, morie-py, hawkes,
   morie-empirical).  Bundle TeX source + compiled PDF, submit to
   stat.ME / cs.MS / cs.LG categories.  First-time submitter
   endorsement may apply.  After acceptance: bump each `Ruhela2026*`
   bib self-cite to v0.4.0-alpha, add `eprint = "26XX.XXXXX",
   archivePrefix = "arXiv"` fields alongside Zenodo DOI.

2. **DEB/RPM packaging** hosted via **GitHub Pages** as a static
   apt/dnf repository.  CI builds .deb + .rpm artifacts on `v0.4.0*`
   tags, publishes to `hadesllm.github.io/morie-repo/`.  Users add
   one apt-repository line + trust the project GPG key.

3. **Deterministic-mode SHA-keyed seed stream** for stochastic
   callables across all 14 textbook suites (Kosorok bootstrap, Deep-
   learning dropout/LSTM, Armstrong Bayesian IRT, Montesinos
   Bayesian/DL, Ghosal MCMC, ML-foundations RF/GBM/xgbst/tsnrd/rndsr).
   Closes the Py↔R bit-portability gap for verification builds.

4. **`morie.entheo` module — Timmermann/HADES DMT-imaging
   integration**.  Beautiful Loop + Self-Aware Networks consciousness
   theories (per `reference_entheo_eeg_fmri_repos.md`).  EEG-fMRI
   preprocessing pipelines for `DMT_Imaging` (Timmermann source, 20
   subj, 15 motion-survived).  Py module `morie.entheo` +
   R parity `morie::entheo_*()`.  Data fetchers reference
   `hadesllm-work/DMT_Imaging/`.

5. **CRAN acceptance** for the morie R package, completing the
   PyPI/r-universe/CRAN trifecta.

Plus org-level polish:
- `hadesllm/.github` profile README + SECURITY.md + CONTRIBUTING.md.
- `hadesllm.r-universe.dev` per-package license + paper DOI display.
