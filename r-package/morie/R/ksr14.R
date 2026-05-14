# SPDX-License-Identifier: MIT OR Apache-2.0

#' Profile likelihood for the linear-regression slope
#'
#' Y = beta X + eps with eps ~ N(0, sigma^2); profiling sigma^2 gives
#' OLS slope with observed-information SE sqrt(sigma2/Sxx).
#'
#' @param x Numeric covariate vector.
#' @param y Numeric outcome vector.
#' @return Named list with estimate, se, n, method.
#' @references Kosorok (2008), Ch 7.
#' @export
ksr14_kosorok_profile_likelihood <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  n <- length(x)
  xc <- x - mean(x); yc <- y - mean(y)
  Sxx <- sum(xc^2)
  beta <- sum(xc * yc) / Sxx
  resid <- yc - beta * xc
  sigma2 <- sum(resid^2) / (n - 2)
  se <- sqrt(sigma2 / Sxx)
  list(
    estimate = beta, se = se, n = n,
    method   = "Profile likelihood for OLS slope (eta=sigma^2 profiled)"
  )
}

# CANONICAL TEST
# set.seed(0); xs <- rnorm(200); ys <- 1.5*xs + rnorm(200)
# ksr14_kosorok_profile_likelihood(xs, ys)

#' @rdname ksr14_kosorok_profile_likelihood
#' @keywords internal
#' @export
kosorok_profile_likelihood <- ksr14_kosorok_profile_likelihood
