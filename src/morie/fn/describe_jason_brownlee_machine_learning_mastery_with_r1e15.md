# describe('jason_brownlee_machine_learning_mastery_with_r1e15') — jason_brownlee_machine_learning_mastery_with_r_chapter_1_equation_15

## WHAT IT DOES

GeneralStatistics equation extracted from Jason Brownlee - Machine Learning Mastery with R.

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

trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic'Character is destiny. — Heraclitus's a RichResult; `isinstance(r, dict)`
  is False but `'statistic' in r` and `r['statistic']` both work.
- Ignoring the warnings/interpretation block when sample sizes are small.
- Confusing this with a similarly-named callable in a different family
  (check `cheatsheet()` for disambiguation).

## REFERENCES

Jason Brownlee - Machine Learning Mastery with R, ch.1 eq.1.15
