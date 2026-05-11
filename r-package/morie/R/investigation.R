#' Run a weighted logistic-regression analysis
#'
#' Mirrors the Python `morie.run_weighted_logistic_analysis()`. Fits a
#' binary-outcome model using survey weights via `survey::svyglm()` if the
#' suggested `survey` package is available, otherwise falls back to base
#' `glm()` with case weights.
#'
#' @param data A `data.frame` containing outcome, predictors, and (optionally)
#'   a weights column.
#' @param outcome Column name of the binary outcome.
#' @param predictors Character vector of predictor column names.
#' @param weights_col Optional column name of analytical weights.
#'
#' @return A list with components `coefficients` (named numeric vector),
#'   `std_errors`, `p_values`, `n`, `method` ("svyglm" or "glm-weighted").
#' @export
#' @examples
#' set.seed(1)
#' df <- data.frame(
#'   y = rbinom(200, 1, 0.4),
#'   x1 = rnorm(200),
#'   x2 = rnorm(200),
#'   w  = runif(200, 0.5, 1.5)
#' )
#' run_weighted_logistic_analysis(df,
#'   outcome = "y", predictors = c("x1", "x2"), weights_col = "w")
run_weighted_logistic_analysis <- function(data, outcome, predictors,
                                           weights_col = NULL) {
  fml <- stats::as.formula(
    paste(outcome, "~", paste(predictors, collapse = " + "))
  )
  use_survey <- !is.null(weights_col) &&
    requireNamespace("survey", quietly = TRUE)

  if (use_survey) {
    design <- survey::svydesign(
      ids = ~1, data = data, weights = stats::as.formula(paste0("~", weights_col))
    )
    fit <- survey::svyglm(fml, design = design,
                          family = stats::quasibinomial())
    method <- "svyglm"
  } else {
    w <- if (!is.null(weights_col)) data[[weights_col]] else NULL
    fit <- stats::glm(fml, data = data, family = stats::binomial(),
                      weights = w)
    method <- if (is.null(w)) "glm-unweighted" else "glm-weighted"
  }
  coef_summary <- stats::coef(summary(fit))
  list(
    coefficients = coef_summary[, "Estimate"],
    std_errors   = coef_summary[, "Std. Error"],
    p_values     = coef_summary[, ncol(coef_summary)],
    n            = stats::nobs(fit),
    method       = method
  )
}

#' Compare nested logistic-regression models via likelihood-ratio test
#'
#' Mirrors the Python `morie.compare_nested_logistic_models()`. Fits a
#' reduced and a full logistic model (the reduced model's predictors must be
#' a subset of the full model's), then performs an analysis-of-deviance LRT.
#'
#' @param data A `data.frame`.
#' @param outcome Column name of the binary outcome.
#' @param predictors_full Character vector: full model's predictors.
#' @param predictors_reduced Character vector: reduced model's predictors.
#'   Must be a subset of `predictors_full`.
#'
#' @return A list with `chi_sq`, `df`, `p_value`, `aic_full`, `aic_reduced`,
#'   `n`.
#' @export
#' @examples
#' set.seed(1)
#' df <- data.frame(
#'   y = rbinom(200, 1, 0.4),
#'   x1 = rnorm(200), x2 = rnorm(200), x3 = rnorm(200)
#' )
#' compare_nested_logistic_models(df,
#'   outcome = "y",
#'   predictors_full    = c("x1", "x2", "x3"),
#'   predictors_reduced = c("x1"))
compare_nested_logistic_models <- function(data, outcome,
                                           predictors_full,
                                           predictors_reduced) {
  if (!all(predictors_reduced %in% predictors_full))
    stop("predictors_reduced must be a subset of predictors_full.",
         call. = FALSE)

  fml_full <- stats::as.formula(
    paste(outcome, "~", paste(predictors_full, collapse = " + "))
  )
  fml_red <- stats::as.formula(
    paste(outcome, "~", paste(predictors_reduced, collapse = " + "))
  )

  fit_full <- stats::glm(fml_full, data = data,
                         family = stats::binomial())
  fit_red  <- stats::glm(fml_red,  data = data,
                         family = stats::binomial())

  chi_sq <- as.numeric(stats::deviance(fit_red) - stats::deviance(fit_full))
  df     <- length(predictors_full) - length(predictors_reduced)
  p_val  <- stats::pchisq(chi_sq, df = df, lower.tail = FALSE)

  list(
    chi_sq      = chi_sq,
    df          = df,
    p_value     = p_val,
    aic_full    = stats::AIC(fit_full),
    aic_reduced = stats::AIC(fit_red),
    n           = stats::nobs(fit_full)
  )
}

#' Run a treatment-effects analysis (point estimate, SE, 95% CI)
#'
#' Mirrors the Python `morie.run_treatment_effects_analysis()`. Convenience
#' wrapper around [estimate_ate()] that also produces a 95% confidence
#' interval (delta-method approximation).
#'
#' @param data A `data.frame`.
#' @param treatment Column name of the binary treatment.
#' @param outcome Column name of the outcome.
#' @param covariates Character vector of covariate column names.
#'
#' @return A list with `ate`, `se`, `ci_lower`, `ci_upper`, `n`, `method`.
#' @export
#' @examples
#' set.seed(1)
#' df <- data.frame(
#'   y = rnorm(200),
#'   t = rbinom(200, 1, 0.5),
#'   x1 = rnorm(200), x2 = rnorm(200)
#' )
#' run_treatment_effects_analysis(df,
#'   treatment = "t", outcome = "y", covariates = c("x1", "x2"))
run_treatment_effects_analysis <- function(data, treatment, outcome,
                                           covariates) {
  ate_res <- estimate_ate(data, treatment = treatment, outcome = outcome,
                          covariates = covariates)
  list(
    ate      = ate_res$ate,
    se       = ate_res$se,
    ci_lower = ate_res$ci_lower,
    ci_upper = ate_res$ci_upper,
    n        = ate_res$n,
    method   = "Hajek IPW ATE (Wald CI)"
  )
}
