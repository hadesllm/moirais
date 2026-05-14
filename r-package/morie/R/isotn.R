# SPDX-License-Identifier: MIT OR Apache-2.0
#' Isotonic regression via PAVA (Barlow et al. 1972)
#'
#' Returns the non-decreasing (or non-increasing) least-squares fit using
#' the pool-adjacent-violators algorithm via stats::isoreg.
#'
#' @param x numeric predictor.
#' @param y numeric outcome.
#' @param weights optional non-negative weights.
#' @param increasing logical (default TRUE).
#' @return list: x_sorted, fitted, residuals, sse, r2, n, method.
#' @keywords internal
isotn <- function(x, y, weights = NULL, increasing = TRUE) {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (n < 2L || length(y) != n)
    return(list(estimate = NA_real_, n = n, method = "Isotonic (n<2)"))
  if (is.null(weights)) weights <- rep(1, n)
  ord <- order(x)
  xs <- x[ord]; ys <- y[ord]; ws <- weights[ord]
  if (increasing) {
    fit <- stats::isoreg(xs, ys)
    fitted <- fit$yf
  } else {
    fit <- stats::isoreg(xs, -ys)
    fitted <- -fit$yf
  }
  resid <- ys - fitted
  sse <- sum(ws * resid^2)
  sst <- sum(ws * (ys - stats::weighted.mean(ys, ws))^2)
  r2 <- if (sst > 0) 1 - sse / sst else NA_real_
  list(x_sorted = xs, fitted = fitted, residuals = resid,
       sse = sse, r2 = as.numeric(r2),
       estimate = mean(fitted), n = as.integer(n),
       method = "Isotonic regression (Barlow et al. 1972, PAVA)")
}

# CANONICAL TEST
# x <- 0:9; y <- c(1, 3, 2, 5, 4, 6, 7, 8, 7, 10)
# r <- isotn(x, y)
# stopifnot(all(diff(r$fitted) >= -1e-9))

#' @rdname isotn
#' @keywords internal
#' @export
isotonic_regression <- isotn
