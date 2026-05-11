# describe('guide_on_data_analysis11u770') — guide_on_data_analysis_chapter_11_unnumbered_770

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

𝑦𝑡 = 𝛽0 + 𝑝𝑒𝑡𝛿0 + 𝑝𝑒𝑡−1𝛿1 + 𝑝𝑒𝑡−2𝛿2 + 𝜖𝑡

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from morie.fn.guide_on_data_analysis11u770 import guide_on_data_analysis_chapter_11_unnumbered_770
import numpy as np
result = guide_on_data_analysis_chapter_11_unnumbered_770(np.random.default_rng(42).normal(0, 1, 100))
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

guide on data analysis, ch.11 (unnumbered)
