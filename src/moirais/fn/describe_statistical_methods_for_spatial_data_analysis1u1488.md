# describe('statistical_methods_for_spatial_data_analysis1u1488') — statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1488

## WHAT IT DOES

Spatial expression (auto-extracted; see ref).

## WHEN TO USE

This callable applies when you have the inputs (x) and want
the outputs (value). See the FORMULA section for the assumed
parametric form.

## WHEN NOT TO USE

- The data violates the formula'Out of chaos, comes order. — Friedrich Nietzsche'statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from moirais.fn.statistical_methods_for_spatial_data_analysis1u1488 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1488
import numpy as np
result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1488(np.random.default_rng(42).normal(0, 1, 100))
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

Schabenberger & Gotway (2005) Statistical Methods for Spatial Data Analysis, ch.1 (unnumbered)
