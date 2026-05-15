# SPDX-License-Identifier: AGPL-3.0-or-later

#' Semiparametric information bound I_eff = Var(X)/sigma^2
#'
#' For the linear model Y = beta X + eps, this is the Fisher-info
#' lower bound on Var(sqrt(n) beta_hat).
#'
#' @param x Numeric covariate vector.
#' @param y Numeric outcome vector.
#' @return Named list with estimate, n, method.
#' @references Kosorok (2008), Ch 6.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ksr12_kosorok_information_bound <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  n <- length(x)
  xc <- x - mean(x); yc <- y - mean(y)
  beta <- sum(xc * yc) / sum(xc^2)
  resid <- yc - beta * xc
  sigma2 <- sum(resid^2) / (n - 2)
  var_x <- sum(xc^2) / n
  list(
    estimate = var_x / sigma2,
    n        = n,
    method   = "Information bound I_eff = Var(X)/sigma^2"
  )
}

# CANONICAL TEST
# set.seed(0); xs <- rnorm(200); ys <- 1.5*xs + rnorm(200)
# ksr12_kosorok_information_bound(xs, ys)

#' @rdname ksr12_kosorok_information_bound
#' @keywords internal
#' @export
kosorok_information_bound <- ksr12_kosorok_information_bound
