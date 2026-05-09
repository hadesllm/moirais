# describe('analysis_of_categorical_data_with_r_chapman_hall_crc_christo3u529') — analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_unnumbered_529

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

log(πj/π1) =βj0 +βj1x1 +βj2x2 +βj3x1x2

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from moirais.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo3u529 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_unnumbered_529
import numpy as np
result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_unnumbered_529(np.random.default_rng(42).normal(0, 1, 100))
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

Analysis of Categorical Data with R (Chapman & Hall CRC -- CHRISTOPHER R   LOUGHIN BILDER (THOMAS M ), ch.3 (unnumbered)
