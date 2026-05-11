# describe('analysis_of_categorical_data_with_r_chapman_hall_crc_christo3u457') — analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_unnumbered_457

## WHAT IT DOES

ContingencyTables expression (auto-extracted; see ref).

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

Below is how we simulate a sample of sizen = 1000 for a2×3 contingency table corresponding to the one multinomial model withπ11 = 0.2, π21 = 0.3, π12 = 0.2, π22 =

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo3u457 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_unnumbered_457
import numpy as np
result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_unnumbered_457(np.random.default_rng(42).normal(0, 1, 100))
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
