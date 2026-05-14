# SPDX-License-Identifier: MIT OR Apache-2.0
#' Importance sampling (Geweke 1989)
#'
#' Given samples x ~ q, estimate E_p of h(X) with weights w = p(x)/q(x).
#' Returns both unnormalised and self-normalised estimators plus ESS.
#'
#' @param x numeric draws from q.
#' @param h,p,q functions; default h(x)=x, p=q=dnorm (sanity-check identity).
#' @return list: estimate, estimate_sn, se, ess, n, method.
#' @keywords internal
impsm <- function(x, h = NULL, p = NULL, q = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 1L)
    return(list(estimate = NA_real_, n = 0L, method = "impsm (empty)"))
  if (is.null(h)) h <- function(z) z
  if (is.null(p)) p <- function(z) stats::dnorm(z)
  if (is.null(q)) q <- function(z) stats::dnorm(z)
  hx <- vapply(x, h, numeric(1))
  px <- vapply(x, p, numeric(1))
  qx <- vapply(x, q, numeric(1))
  w  <- px / qx
  est    <- mean(w * hx)
  est_sn <- sum(w * hx) / sum(w)
  se     <- stats::sd(w * hx) / sqrt(n)
  ess    <- sum(w)^2 / sum(w^2)
  list(estimate = as.numeric(est), estimate_sn = as.numeric(est_sn),
       se = as.numeric(se), ess = as.numeric(ess), n = as.integer(n),
       method = "Importance sampling (Geweke 1989)")
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(5000)
# r <- impsm(x, h = function(z) z^2)
# # E[X^2] under N(0,1) = 1
# stopifnot(abs(r$estimate - 1) < 0.2)

#' @rdname impsm
#' @keywords internal
#' @export
importance_sampling <- impsm
