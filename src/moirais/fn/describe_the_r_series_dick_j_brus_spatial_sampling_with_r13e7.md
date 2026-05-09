# describe('the_r_series_dick_j_brus_spatial_sampling_with_r13e7') — the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_7

## WHAT IT DOES

Dispersion equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.

## WHEN TO USE

This callable applies when you have the inputs (x) and want
the outputs (value). See the FORMULA section for the assumed
parametric form.

## WHEN NOT TO USE

- The data violates the formula's domain assumptions (e.g. zero variance,
  perfectly collinear inputs).
- A more specialised version of this method exists for your data shape.
- Sample size is too small for the asymptotics this estimator relies on.

## ASSUMPTIONS

- Inputs are real-valued and free of NaN/Inf.
- Observations are independent unless the method explicitly handles
  clustering.
- Distributional assumptions vary; see the formula and reference for
  specifics.

## FORMULA

equal size, 𝑤ℎ = 1/𝑛, and that 𝑛ℎ = 1. Therefore, Equation ( 13.5) reduces to

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from moirais.fn.the_r_series_dick_j_brus_spatial_sampling_with_r13e7 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_7
import numpy as np
result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_7(np.random.default_rng(42).normal(0, 1, 100))
print(result)              # full multi-section render
result.payload             # raw dict
```

## COMMON MISTAKES

- Treating the result as a plain dict — it's a RichResult; `isinstance(r, dict)`
  is False but `'statistic' in r` and `r['statistic']` both work.
- Ignoring the warnings/interpretation block when sample sizes are small.
- Confusing this with a similarly-named callable in a different family
  (check `cheatsheet()` for disambiguation).

## REFERENCES

[The R Series] Dick J. Brus - Spatial Sampling with R, ch.13 eq.13.7
