# SPDX-License-Identifier: GPL-2.0-only

#' BvM diagnostic for the mean functional under a DP prior.
#'
#' @param x Numeric data vector.
#' @param theta0 Optional null value for the mean functional.
#' @param B Integer number of bootstrap draws (default 500).
#' @param seed Integer RNG seed (default 0).
#' @return Named list with estimate, se, theta_hat, z_ks_stat, z_ks_pvalue,
#'   wald, wald_pvalue, n, B, method.
#' @export
ghosal_bernstein_von_mises <- function(x, theta0 = NULL, B = 500, seed = 0) {
  set.seed(seed)
  x <- as.numeric(x); n <- length(x)
  if (n < 2) return(list(estimate = NA_real_, se = NA_real_, n = n,
                          method = "BvM (n<2)"))
  theta_hat <- mean(x); s <- sd(x)
  draws <- numeric(B)
  for (b in seq_len(B)) {
    g <- stats::rgamma(n, 1, 1); u <- g / sum(g)
    draws[b] <- sum(u * x)
  }
  z <- (draws - theta_hat) * sqrt(n) / max(s, 1e-12)
  ks <- suppressWarnings(stats::ks.test(z, "pnorm"))
  if (!is.null(theta0)) {
    sd_d <- sd(draws); wald <- (mean(draws) - theta0) / max(sd_d, 1e-12)
    wald_p <- 2 * (1 - stats::pnorm(abs(wald)))
  } else {
    wald <- NA_real_; wald_p <- NA_real_
  }
  list(estimate = mean(draws), se = sd(draws), theta_hat = theta_hat,
       z_ks_stat = unname(ks$statistic), z_ks_pvalue = ks$p.value,
       wald = wald, wald_pvalue = wald_p, n = n, B = B,
       method = "BvM for mean functional (Bayesian bootstrap)")
}
