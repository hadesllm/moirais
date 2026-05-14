# SPDX-License-Identifier: MIT OR Apache-2.0

#' Fauzi: Kernel survival function estimator (Ch 4)
#'
#' \eqn{\hat S_h(t)=1-\hat F_h(t)} with asymptotic 95\% CI.
#'
#' @param x Numeric vector (lifetimes).
#' @param t Evaluation point; default median(x).
#' @param h Bandwidth; default = Silverman.
#' @return Named list with estimate, se, ci_lower, ci_upper, t, h, n, method.
#' @importFrom stats median pnorm
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
fzsrv <- function(x, t = NULL, h = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2L) return(list(estimate = NA_real_, n = n,
                           method = "fzsrv - too few obs"))
  if (is.null(t)) t <- stats::median(x)
  if (is.null(h)) h <- .morie_silverman_h(x)
  F_hat <- mean(stats::pnorm((t - x) / h))
  S_hat <- 1 - F_hat
  se <- sqrt(S_hat * (1 - S_hat) / n)
  z <- 1.959963984540054
  list(estimate = S_hat, se = se,
       ci_lower = max(0, S_hat - z * se),
       ci_upper = min(1, S_hat + z * se),
       t = t, h = h, n = n,
       method = "Fauzi kernel survival S_hat(t)=1-F_hat_h(t) (Ch 4)")
}

# CANONICAL TEST
# set.seed(0); x <- rexp(2000, 1)
# r <- fzsrv(x, t = 1); stopifnot(abs(r$estimate - exp(-1)) < 0.05)

#' @rdname fzsrv
#' @keywords internal
#' @export
fauzi_survival_kernel <- fzsrv
