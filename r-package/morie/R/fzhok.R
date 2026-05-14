# SPDX-License-Identifier: MIT OR Apache-2.0

#' Fauzi: Higher-order (order-4) Gaussian-based kernel (Ch 1)
#'
#' KDE with order-4 Wand-Jones (1995, eq 2.8) kernel
#' \eqn{K_4(u) = (1/2)(3-u^2)\phi(u)}.  Bias reduces from O(h^2)
#' to O(h^4).  Note: K_4 takes negative values so f_hat may be < 0.
#'
#' @param x Numeric vector.
#' @param t Evaluation point; default median(x).
#' @param h Bandwidth; default = Silverman.
#' @param order Kernel order (only 4 supported).
#' @return Named list with estimate, h, t, order, mu_r, R_K, n, method.
#' @importFrom stats median dnorm
#' @export
fzhok <- function(x, t = NULL, h = NULL, order = 4L) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2L) return(list(estimate = NA_real_, n = n,
                           method = "fzhok - too few obs"))
  if (order != 4L) stop("only order=4 implemented")
  if (is.null(t)) t <- stats::median(x)
  if (is.null(h)) h <- .morie_silverman_h(x)
  u <- (t - x) / h
  k4 <- 0.5 * (3 - u^2) * stats::dnorm(u)
  f_hat <- mean(k4) / h
  list(estimate = f_hat, h = h, t = t, order = order,
       mu_r = -3, R_K = 27 / (32 * sqrt(pi)), n = n,
       method = "Fauzi higher-order (4) kernel density (Ch 1)")
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(2000); r <- fzhok(x, t = 0)
# stopifnot(abs(r$estimate - dnorm(0)) < 0.1)

#' @rdname fzhok
#' @keywords internal
#' @export
fauzi_higher_order_kernel <- fzhok
