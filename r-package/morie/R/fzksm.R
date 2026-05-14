# SPDX-License-Identifier: MIT OR Apache-2.0

#' Fauzi: Kolmogorov-Smirnov test with kernel-smoothed CDF (Ch 5)
#'
#' \eqn{D_n = \sup_t |\hat F_h(t) - F_0(t)|}.
#'
#' @param x Numeric vector.
#' @param cdf Either a SciPy-style CDF name (one of "norm" only here;
#'   pass a function for anything else) or a function F0(t).
#' @param args List of distribution args (default = MLE for "norm").
#' @param h Bandwidth; default = Silverman.
#' @param n_grid Grid resolution.
#' @return Named list: statistic, p_value, h, n, method.
#' @importFrom stats sd pnorm
#' @export
fzksm <- function(x, cdf = "norm", args = NULL, h = NULL, n_grid = 512L) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5L) return(list(statistic = NA_real_, p_value = NA_real_, n = n,
                           method = "fzksm - too few obs"))
  if (is.null(h)) h <- .morie_silverman_h(x)
  if (is.function(cdf)) {
    F0 <- cdf
  } else if (identical(cdf, "norm")) {
    if (is.null(args)) args <- list(mean(x), stats::sd(x))
    F0 <- function(t) stats::pnorm(t, mean = args[[1]], sd = args[[2]])
  } else {
    stop("supply a function for non-normal cdf")
  }
  lo <- min(x) - 6 * h; hi <- max(x) + 6 * h
  grid <- seq(lo, hi, length.out = n_grid)
  F_hat <- vapply(grid, function(g) mean(stats::pnorm((g - x) / h)),
                  numeric(1))
  F_ref <- vapply(grid, F0, numeric(1))
  D_n <- max(abs(F_hat - F_ref))
  # Kolmogorov asymptotic tail:  P(K > x) = 2 * sum_{k>=1} (-1)^(k-1) exp(-2 k^2 x^2)
  lam <- sqrt(n) * D_n
  k <- 1:100
  pval <- 2 * sum((-1)^(k - 1) * exp(-2 * k^2 * lam^2))
  pval <- max(0, min(1, pval))
  list(statistic = D_n, p_value = pval, h = h, n = n,
       method = "Fauzi kernel-smoothed KS test (Ch 5)")
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(500)
# r <- fzksm(x, cdf = "norm", args = list(0, 1)); stopifnot(r$p_value > 0.05)

#' @rdname fzksm
#' @keywords internal
#' @export
fauzi_ks_smoothed <- fzksm
