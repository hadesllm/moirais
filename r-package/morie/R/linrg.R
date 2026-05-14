# SPDX-License-Identifier: MIT OR Apache-2.0

#' Ordinary least squares closed-form solution (R parity)
#'
#' Wraps \code{stats::lm} and returns coefficients plus classical OLS
#' standard errors.
#'
#' @param x Numeric matrix or vector of predictors.
#' @param y Numeric response vector.
#' @return Named list with \code{estimate} (intercept + slopes),
#'   \code{se}, \code{n}, \code{method}.
#' @references
#' Hastie, Tibshirani & Friedman, Elements of Statistical Learning (2009).
#' @export
linear_regression_ols <- function(x, y) {
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); y <- as.numeric(y)
  df <- as.data.frame(x); df$.y <- y
  fit <- stats::lm(.y ~ ., data = df)
  s <- summary(fit)
  est <- unname(stats::coef(fit))
  se  <- unname(s$coefficients[, "Std. Error"])
  list(
    estimate = est,
    se       = se,
    n        = nrow(x),
    method   = "OLS via closed-form normal equations"
  )
}
