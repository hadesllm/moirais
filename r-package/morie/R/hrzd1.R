# SPDX-License-Identifier: GPL-2.0-only

#' Cox partial-likelihood proportional-hazards estimator
#'
#' @param t Numeric vector of observed event/censoring times.
#' @param x Numeric covariate vector or design matrix.
#' @param event Integer/logical vector (1 = event, 0 = censored).
#' @return Named list with estimate, se, n, n_events, method.
#' @keywords internal
hrzd1 <- function(t, x, event) {
  t <- as.numeric(t); event <- as.numeric(event)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p) || length(t) != n || length(event) != n)
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "Cox PH (insufficient data)"))
  o <- order(-t)
  Xs <- X[o, , drop = FALSE]; ev <- event[o]
  beta <- rep(0, p)
  H <- diag(p)
  for (it in 1:50) {
    eta <- pmin(pmax(as.numeric(Xs %*% beta), -50), 50)
    ehb <- exp(eta)
    S0 <- cumsum(ehb)
    S1 <- apply(Xs * ehb, 2, cumsum)
    if (is.null(dim(S1))) S1 <- matrix(S1, ncol = p)
    mean_X <- S1 / pmax(S0, 1e-12)
    diff_X <- Xs - mean_X
    score <- colSums(ev * diff_X)
    # build cumulative S2 (n, p, p)
    S2 <- array(0, c(n, p, p))
    for (i in 1:p) for (j in 1:p) {
      S2[, i, j] <- cumsum(Xs[, i] * Xs[, j] * ehb)
    }
    var_X <- array(0, c(n, p, p))
    for (i in 1:p) for (j in 1:p) {
      var_X[, i, j] <- S2[, i, j] / pmax(S0, 1e-12) - mean_X[, i] * mean_X[, j]
    }
    info <- matrix(0, p, p)
    for (i in 1:p) for (j in 1:p) info[i, j] <- sum(ev * var_X[, i, j])
    step <- tryCatch(solve(info + 1e-8 * diag(p), score),
                     error = function(e) MASS::ginv(info) %*% score)
    beta <- beta + as.numeric(step)
    if (max(abs(step)) < 1e-6) break
  }
  cov_m <- tryCatch(MASS::ginv(info), error = function(e) matrix(NA, p, p))
  se <- sqrt(pmax(diag(cov_m), 0))
  list(estimate = if (p == 1) as.numeric(beta) else as.numeric(beta),
       se = if (p == 1) as.numeric(se) else as.numeric(se),
       n = n, n_events = as.integer(sum(event)),
       method = "Cox proportional hazards (partial likelihood)")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzd1
#' @keywords internal
#' @export
horowitz_duration_model <- hrzd1
