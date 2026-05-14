# SPDX-License-Identifier: MIT OR Apache-2.0
#' Leave-one-out cross-validation for ordinary kriging.
#'
#' MSPE = (1/n) sum (Z(s_i) - Z_hat_minus_i(s_i))^2.
#'
#' @param x Numeric vector.
#' @param coords Coord matrix.
#' @param nugget,sill,range_ Exponential-covariance parameters.
#' @return Named list: estimate (MSPE, RMSPE, MAE, residuals), n, method.
#' @references Schabenberger & Gotway (2005), Ch 4.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
spcrs <- function(x, coords, nugget = 0, sill = 1, range_ = 1) {
  x <- as.numeric(x); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  c0 <- nugget; c1 <- sill - nugget; a <- range_
  D <- as.matrix(stats::dist(coords))
  cov_fn <- function(h) c1 * exp(-h / a) + ifelse(h == 0, c0, 0)
  resid <- numeric(n)
  for (i in seq_len(n)) {
    sel <- setdiff(seq_len(n), i)
    m <- n - 1
    Cii <- cov_fn(D[sel, sel])
    c_vec <- cov_fn(D[i, sel])
    A <- matrix(0, m + 1, m + 1)
    A[1:m, 1:m] <- Cii; A[1:m, m + 1] <- 1; A[m + 1, 1:m] <- 1
    rhs <- c(c_vec, 1)
    sol <- tryCatch(solve(A, rhs),
                    error = function(e) qr.solve(A, rhs))
    z_hat <- sum(sol[1:m] * x[sel])
    resid[i] <- x[i] - z_hat
  }
  mspe <- mean(resid ^ 2)
  list(estimate = list(MSPE = mspe, RMSPE = sqrt(mspe),
                       MAE = mean(abs(resid)), residuals = resid),
       n = n, method = "LOO cross-validation for ordinary kriging")
}

#' @rdname spcrs
#' @keywords internal
#' @export
spatial_cross_validation <- spcrs
