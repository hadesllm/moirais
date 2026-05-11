# describe('andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp5e2') — andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_5_equation_2

## WHAT IT DOES

CountModels equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.

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

ri = yi − 1

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp5e2 import andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_5_equation_2
import numpy as np
result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_5_equation_2(np.random.default_rng(42).normal(0, 1, 100))
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

Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling, ch.5 eq.5.2
