# SPDX-License-Identifier: MIT OR Apache-2.0

#' Influence function of OLS beta-hat
#'
#' IF(x, y) = (y - ybar - beta_hat (x - xbar))(x - xbar) / Var(X).
#'
#' @param x Numeric covariate vector.
#' @param y Numeric outcome vector.
#' @return Named list with estimate, n, method.
#' @references Kosorok (2008), Ch 7.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ksr16_kosorok_influence_function <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  n <- length(x)
  xc <- x - mean(x); yc <- y - mean(y)
  var_x <- sum(xc^2) / n
  beta <- sum(xc * yc) / sum(xc^2)
  resid <- yc - beta * xc
  IF <- (resid * xc) / var_x
  list(
    estimate = mean(IF),
    n        = n,
    method   = "Influence function: (resid)(x-xbar)/Var(X)"
  )
}

# CANONICAL TEST
# set.seed(0); xs <- rnorm(200); ys <- 1.5*xs + rnorm(200)
# ksr16_kosorok_influence_function(xs, ys)

#' @rdname ksr16_kosorok_influence_function
#' @keywords internal
#' @export
kosorok_influence_function <- ksr16_kosorok_influence_function
