# SPDX-License-Identifier: GPL-2.0-only

#' Maximal inequality bound for empirical processes
#'
#' E* of sup |G_n(f)| <= J_bracket(theta_n, F, L_2(P)) * sigma_n for the
#' indicator class with theta_n = 0.5.
#'
#' @param x Numeric vector.
#' @return Named list with estimate, n, method.
#' @references Kosorok (2008), Ch 2.
#' @export
ksr06_kosorok_maximal_inequality <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  sigma_n <- if (n > 1L) stats::sd(x) else NA_real_
  theta_n <- 0.5
  integrand <- function(e) sqrt(log(2) - 2 * log(e))
  j <- stats::integrate(integrand, lower = 1e-8, upper = theta_n,
                        subdivisions = 200L)$value
  list(
    estimate = j * sigma_n,
    n        = n,
    method   = "Maximal-inequality RHS: J_[](theta_n) * sigma_n"
  )
}

# CANONICAL TEST
# set.seed(0); ksr06_kosorok_maximal_inequality(rnorm(200))

#' @rdname ksr06_kosorok_maximal_inequality
#' @keywords internal
#' @export
kosorok_maximal_inequality <- ksr06_kosorok_maximal_inequality
