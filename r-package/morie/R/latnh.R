# SPDX-License-Identifier: MIT OR Apache-2.0
#' Latin hypercube sampling (McKay, Beckman & Conover 1979)
#'
#' Stratified random sample on the unit cube of dimension d: each dimension is split into N
#' equal-probability strata, one sample per stratum, then strata are
#' permuted independently across dimensions.
#'
#' @param N integer; sample size (default 100).
#' @param d integer; dimensionality (default 1).
#' @param f optional integrand on (0,1)^d returning scalar.
#' @param seed integer.
#' @return list: sample (N x d matrix), estimate (if f given), se, N, d, method.
#' @keywords internal
latnh <- function(N = 100L, d = 1L, f = NULL, seed = 42L) {
  set.seed(seed)
  cut <- (seq_len(N) - 1) / N
  sample <- matrix(0, nrow = N, ncol = d)
  for (j in seq_len(d)) {
    u_j <- cut + stats::runif(N, 0, 1 / N)
    sample[, j] <- sample(u_j)
  }
  out <- list(sample = sample, N = as.integer(N), d = as.integer(d),
              method = "Latin hypercube (McKay et al. 1979)")
  if (!is.null(f)) {
    fv <- apply(sample, 1, f)
    out$estimate <- mean(fv)
    out$se <- stats::sd(fv) / sqrt(N)
  }
  out
}

# CANONICAL TEST
# r <- latnh(N = 500, d = 2, f = function(u) u[1] + u[2], seed = 0)
# stopifnot(abs(r$estimate - 1) < 0.05)

#' @rdname latnh
#' @keywords internal
#' @export
latin_hypercube <- latnh
