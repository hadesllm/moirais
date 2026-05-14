# SPDX-License-Identifier: MIT OR Apache-2.0

#' Z-estimator asymptotic distribution
#'
#' theta_n solves P_n psi(.; theta) = 0 with sandwich variance
#' V = solve(A) %*% B %*% t(solve(A)).
#'
#' @param x Numeric vector.
#' @param y Optional numeric vector; if supplied returns OLS slope.
#' @return Named list with estimate, se, n, method.
#' @references Kosorok (2008), Ch 5.
#' @export
ksr09_kosorok_z_estimator <- function(x, y = NULL) {
  x <- as.numeric(x)
  if (is.null(y)) {
    n <- length(x)
    theta <- mean(x)
    psi <- x - theta
    se  <- sqrt(mean(psi^2) / n)
    list(estimate = theta, se = se, n = n,
         method = "Z-estimator: psi(x;theta) = x - theta")
  } else {
    y <- as.numeric(y)
    n <- length(x)
    xc <- x - mean(x); yc <- y - mean(y)
    beta <- sum(xc * yc) / sum(xc^2)
    resid <- yc - beta * xc
    A <- mean(xc^2)
    B <- mean((xc^2) * (resid^2))
    se <- sqrt(B / (A^2) / n)
    list(estimate = beta, se = se, n = n,
         method = "Z-estimator: psi(x,y;beta) = x(y - beta x)")
  }
}

# CANONICAL TEST
# set.seed(0); xs <- rnorm(200); ys <- 1.5*xs + rnorm(200)
# ksr09_kosorok_z_estimator(xs, ys)

#' @rdname ksr09_kosorok_z_estimator
#' @keywords internal
#' @export
kosorok_z_estimator <- ksr09_kosorok_z_estimator
