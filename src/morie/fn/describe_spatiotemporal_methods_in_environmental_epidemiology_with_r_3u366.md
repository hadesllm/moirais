# describe('spatiotemporal_methods_in_environmental_epidemiology_with_r_3u366') — spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_366

## WHAT IT DOES

Correlation expression (auto-extracted; see ref).

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

where log(ψv) = a0 + a1v + a2v2 and log(φv) = b0 + b1v + b2v2. The correlation of

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from morie.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_3u366 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_366
import numpy as np
result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_366(np.random.default_rng(42).normal(0, 1, 100))
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

spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818, ch.3 (unnumbered)
