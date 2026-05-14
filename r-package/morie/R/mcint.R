# SPDX-License-Identifier: MIT OR Apache-2.0
#' Monte Carlo integration (Rubinstein 1981)
#'
#' Estimates I = int_a^b f(u) du as I_hat = (b - a) * mean(f(U_i)) with
#' U_i iid Uniform(a, b).  Returns the MC SE.  The corresponding Python
#' callable (\code{morie.fn.mcint}) retains a legacy *spatial* signature;
#' this R parity targets the index-listed crude-MC estimator instead.
#'
#' @param f function on (a, b).
#' @param a,b numeric scalars (bounds).  Default (0, 1).
#' @param N integer; sample size (default 1000).
#' @param seed integer.
#' @return Named list: estimate, se, N, method.
#' @keywords internal
mcint_crude <- function(f, a = 0, b = 1, N = 1000L, seed = 42L) {
  set.seed(seed)
  u <- stats::runif(N, a, b)
  fu <- vapply(u, f, numeric(1))
  est <- (b - a) * mean(fu)
  se  <- (b - a) * stats::sd(fu) / sqrt(N)
  list(estimate = as.numeric(est), se = as.numeric(se),
       N = as.integer(N), method = "Monte Carlo integration (Rubinstein 1981)")
}

# CANONICAL TEST
# r <- mcint_crude(function(u) u^2, 0, 1, N = 5000, seed = 0)
# # int_0^1 u^2 du = 1/3
# stopifnot(abs(r$estimate - 1/3) < 0.02)

#' @rdname mcint_crude
#' @keywords internal
#' @export
monte_carlo_integration <- mcint_crude
