# describe('the_r_series_dick_j_brus_spatial_sampling_with_r11u63') — the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_63

## WHAT IT DOES

Regression expression (auto-extracted; see ref).

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

[EQ] P = I − X(XTX)−1XT , (21.27)

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from moirais.fn.the_r_series_dick_j_brus_spatial_sampling_with_r11u63 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_63
import numpy as np
result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_63(np.random.default_rng(42).normal(0, 1, 100))
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

[The R Series] Dick J. Brus - Spatial Sampling with R, ch.11 (unnumbered)
