# describe('advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th10u60') — advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_60

## WHAT IT DOES

GeneralStatistics expression (auto-extracted; see ref).

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

pairlist = flat_map_chr(as.list(x), find_assign_rec),

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from moirais.fn.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th10u60 import advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_60
import numpy as np
result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_60(np.random.default_rng(42).normal(0, 1, 100))
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

Advanced R (Second Edition) -- Hadley Wickham -- Chapman & Hall CRC the R, ch.10 (unnumbered)
