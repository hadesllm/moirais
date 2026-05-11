test_that("calculate_ebac respects Widmark formula and floors at zero", {
  # Drinks=4, weight=180lb, hours=2, gender=0.73 (male) → ~0.126
  ebac <- calculate_ebac(drinks = 4, weight_lbs = 180,
                         hours = 2, gender_constant = 0.73)
  expect_type(ebac, "double")
  expect_true(ebac > 0 && ebac < 0.5)
  # 8h after 1 drink: should floor at 0
  expect_equal(calculate_ebac(1, 180, 8, 0.73), 0)
  # Zero/negative weight defends:
  expect_equal(calculate_ebac(2, 0, 1, 0.73), 0)
})

test_that("is_over_legal_limit returns 0/1 integers", {
  expect_identical(is_over_legal_limit(0.05), 0L)
  expect_identical(is_over_legal_limit(0.09), 1L)
  expect_identical(is_over_legal_limit(0.05, limit = 0.05), 1L)
  expect_identical(is_over_legal_limit(0.04, limit = 0.05), 0L)
})

test_that("calculate_ipw_weights produces correct length and clipping", {
  set.seed(1)
  df <- data.frame(
    t  = rbinom(100, 1, 0.4),
    ps = pmin(pmax(runif(100, 0.05, 0.95), 0.05), 0.95)
  )
  w <- calculate_ipw_weights(df, treatment = "t", ps_col = "ps")
  expect_length(w, 100)
  expect_true(all(w > 0))
  # stabilized version
  ws <- calculate_ipw_weights(df, treatment = "t", ps_col = "ps",
                              stabilized = TRUE)
  expect_length(ws, 100)
  # Stabilized weights should have smaller variance than unstabilized
  expect_lt(stats::var(ws), stats::var(w) + 1e-9)
  # Trim quantiles
  wt <- calculate_ipw_weights(df, treatment = "t", ps_col = "ps",
                              trim_quantiles = c(0.05, 0.95))
  expect_lte(max(wt), max(w))
})

test_that("calculate_ipw_weights validates trim_quantiles length", {
  df <- data.frame(t = c(1, 0), ps = c(0.5, 0.5))
  expect_error(
    calculate_ipw_weights(df, "t", "ps", trim_quantiles = 0.5),
    "length 2"
  )
})

test_that("infer_measurement_level handles standard cases", {
  expect_equal(infer_measurement_level(c(0, 1, 1, 0)), "binary")
  expect_equal(infer_measurement_level(c(TRUE, FALSE)), "binary")
  expect_equal(infer_measurement_level(factor(c("a", "b", "c"))), "nominal")
  expect_equal(
    infer_measurement_level(ordered(c("low", "med", "high"))),
    "ordinal"
  )
  expect_equal(infer_measurement_level(c(1.2, 3.4, 5.6)), "ratio")
  expect_equal(infer_measurement_level(c(-1.5, 0.0, 2.3)), "interval")
})

test_that("profile_dataset returns expected shape", {
  p <- profile_dataset(iris)
  expect_named(p, c("n_rows", "n_cols", "columns"))
  expect_equal(p$n_rows, nrow(iris))
  expect_equal(p$n_cols, ncol(iris))
  expect_named(p$columns, names(iris))
  # Numeric columns should have summary stats
  expect_true(all(c("mean", "sd", "min", "max", "q25", "q50", "q75") %in%
                  names(p$columns$Sepal.Length)))
  # Factor column should have measurement_level = nominal
  expect_equal(p$columns$Species$measurement_level, "nominal")
})

test_that("profile_dataset rejects non-data.frame inputs", {
  expect_error(profile_dataset(1:10), "data.frame")
})

test_that("suggest_analysis_plan returns character recommendations", {
  s <- suggest_analysis_plan(profile_dataset(iris))
  expect_type(s, "character")
  expect_true(length(s) >= 1L)
})

test_that("suggest_analysis_plan flags missingness", {
  df <- iris
  df$Sepal.Length[1] <- NA
  s <- suggest_analysis_plan(profile_dataset(df))
  expect_true(any(grepl("[Mm]issing", s)))
})

test_that("compare_nested_logistic_models returns LRT components", {
  set.seed(1)
  df <- data.frame(
    y  = rbinom(200, 1, 0.4),
    x1 = rnorm(200), x2 = rnorm(200), x3 = rnorm(200)
  )
  res <- compare_nested_logistic_models(
    df, outcome = "y",
    predictors_full    = c("x1", "x2", "x3"),
    predictors_reduced = c("x1")
  )
  expect_named(res, c("chi_sq", "df", "p_value", "aic_full",
                      "aic_reduced", "n"))
  expect_equal(res$df, 2L)
  expect_true(res$p_value >= 0 && res$p_value <= 1)
  expect_true(is.finite(res$chi_sq))
})

test_that("compare_nested_logistic_models rejects non-nested input", {
  df <- data.frame(y = rbinom(50, 1, 0.5), a = rnorm(50), b = rnorm(50))
  expect_error(
    compare_nested_logistic_models(df, "y",
                                    predictors_full    = c("a"),
                                    predictors_reduced = c("b")),
    "subset"
  )
})

test_that("run_treatment_effects_analysis returns ATE shape", {
  set.seed(1)
  df <- data.frame(
    y = rnorm(200), t = rbinom(200, 1, 0.5),
    x1 = rnorm(200), x2 = rnorm(200)
  )
  res <- run_treatment_effects_analysis(
    df, treatment = "t", outcome = "y", covariates = c("x1", "x2")
  )
  expect_named(res, c("ate", "se", "ci_lower", "ci_upper", "n", "method"))
  expect_true(is.finite(res$ate))
  expect_true(res$ci_lower < res$ci_upper)
})

test_that("run_weighted_logistic_analysis returns coefficient table", {
  set.seed(1)
  df <- data.frame(
    y = rbinom(200, 1, 0.4),
    x1 = rnorm(200), x2 = rnorm(200),
    w  = runif(200, 0.5, 1.5)
  )
  res <- run_weighted_logistic_analysis(
    df, outcome = "y", predictors = c("x1", "x2"), weights_col = "w"
  )
  expect_named(res, c("coefficients", "std_errors", "p_values", "n", "method"))
  # 3 coefs: (Intercept), x1, x2
  expect_length(res$coefficients, 3L)
  expect_equal(res$n, 200)
})

test_that("inspect_output handles JSON/CSV/RDS extensions", {
  skip_if_not_installed("jsonlite")
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp), add = TRUE)
  jsonlite::write_json(list(estimate = 0.123, se = 0.045), tmp)
  res <- inspect_output(tmp)
  expect_true(res$exists)
  expect_equal(res$format, "json")
  expect_equal(res$status, "ok")

  tmp2 <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp2), add = TRUE)
  utils::write.csv(iris, tmp2, row.names = FALSE)
  res2 <- inspect_output(tmp2)
  expect_true(res2$exists)
  expect_equal(res2$format, "csv")
})

test_that("inspect_output reports missing files", {
  res <- inspect_output(tempfile(fileext = ".json"))
  expect_false(res$exists)
  expect_equal(res$status, "missing")
})

test_that("verify_statistical_output runs sanity checks", {
  skip_if_not_installed("jsonlite")
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp), add = TRUE)
  jsonlite::write_json(
    list(ate = 0.5, se = 0.1, ci_lower = 0.3, ci_upper = 0.7,
         n = 200, p_value = 0.001),
    tmp, auto_unbox = TRUE
  )
  res <- verify_statistical_output(tmp)
  expect_true(res$passed)
  expect_true(res$checks$se_nonneg)
  expect_true(res$checks$ci_ordered)
  expect_true(res$checks$estimate_in_ci)
  expect_true(res$checks$n_positive)
  expect_true(res$checks$p_in_unit)
})

test_that("verify_statistical_output flags inverted CI", {
  skip_if_not_installed("jsonlite")
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp), add = TRUE)
  jsonlite::write_json(
    list(ate = 0.5, se = 0.1, ci_lower = 0.7, ci_upper = 0.3, n = 200),
    tmp, auto_unbox = TRUE
  )
  res <- verify_statistical_output(tmp)
  expect_false(res$passed)
  expect_false(res$checks$ci_ordered)
})

test_that("estimate_irm errors informatively when DoubleML missing", {
  # When DoubleML is installed locally we skip; otherwise verify the
  # error path. Either way, this just confirms the gate behaviour.
  if (!requireNamespace("DoubleML", quietly = TRUE)) {
    df <- data.frame(y = rnorm(50), t = rbinom(50, 1, 0.5),
                     x1 = rnorm(50), x2 = rnorm(50))
    expect_error(
      estimate_irm(df, "t", "y", c("x1", "x2")),
      "DoubleML|mlr3"
    )
  } else {
    succeed()
  }
})
