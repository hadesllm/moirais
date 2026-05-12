# SPDX-License-Identifier: GPL-2.0-only

#' Gaussian multiplier bootstrap for Z-estimators
#'
#' G_n_xi(f) = n raised to the power of -1/2 times sum_i xi_i (f(X_i) - P_n f), xi ~ N(0,1).
#'
#' @param x Numeric vector.
#' @param B Number of multiplier replications.
#' @param seed Integer RNG seed.
#' @return Named list with estimate, se, n, method.
#' @references Kosorok (2008), Ch 10.
#' @export
ksr08_kosorok_multiplier_bootstrap <- function(x, B = 1000, seed = 0) {
  x <- as.numeric(x)
  n <- length(x)
  set.seed(seed)
  pn <- mean(x)
  centred <- x - pn
  xi <- matrix(stats::rnorm(B * n), nrow = B)
  g_xi <- (xi %*% centred) / sqrt(n)
  list(
    estimate = mean(g_xi),
    se       = stats::sd(as.numeric(g_xi)),
    n        = n,
    method   = "Multiplier bootstrap G_n^xi = n^{-1/2} sum xi (f-Pf)"
  )
}

# CANONICAL TEST
# set.seed(0); ksr08_kosorok_multiplier_bootstrap(rnorm(200), B=500, seed=42)

#' @rdname ksr08_kosorok_multiplier_bootstrap
#' @keywords internal
#' @export
kosorok_multiplier_bootstrap <- ksr08_kosorok_multiplier_bootstrap
