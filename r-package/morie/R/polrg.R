# SPDX-License-Identifier: GPL-2.0-only

#' Polynomial regression (R parity)
#'
#' Polynomial feature expansion + OLS via \code{stats::poly} +
#' \code{stats::lm}.  Uses raw (not orthogonal) polynomials for parity
#' with scikit-learn's PolynomialFeatures.
#'
#' @param x Numeric vector or matrix.
#' @param y Numeric response.
#' @param degree Polynomial degree.
#' @return Named list: estimate, se, feature_names, degree, n, method.
#' @export
polynomial_regression <- function(x, y, degree = 2L) {
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); y <- as.numeric(y)
  n <- nrow(x); p <- ncol(x)
  cols <- list()
  names_ <- character()
  for (j in seq_len(p)) {
    for (d in seq_len(degree)) {
      cols[[length(cols) + 1L]] <- x[, j]^d
      names_ <- c(names_, paste0("x", j - 1L, if (d > 1L) paste0("^", d) else ""))
    }
  }
  # Add cross-terms only when p > 1 and degree >= 2 (parity-ish with sklearn)
  if (p > 1L && degree >= 2L) {
    combos <- utils::combn(p, 2)
    for (k in seq_len(ncol(combos))) {
      i1 <- combos[1, k]; i2 <- combos[2, k]
      cols[[length(cols) + 1L]] <- x[, i1] * x[, i2]
      names_ <- c(names_, paste0("x", i1 - 1L, " x", i2 - 1L))
    }
  }
  Xp <- do.call(cbind, cols)
  colnames(Xp) <- names_
  df <- as.data.frame(Xp); df$.y <- y
  fit <- stats::lm(.y ~ ., data = df)
  s <- summary(fit)
  list(
    estimate      = unname(stats::coef(fit)),
    se            = unname(s$coefficients[, "Std. Error"]),
    feature_names = c("(intercept)", names_),
    degree        = as.integer(degree),
    n             = n,
    method        = sprintf("Polynomial regression (degree=%d)", as.integer(degree))
  )
}
