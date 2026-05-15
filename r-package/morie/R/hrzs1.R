# SPDX-License-Identifier: AGPL-3.0-or-later

#' Heckman-Powell-Newey-Vella semiparametric sample-selection
#'
#' @param x Numeric outcome covariates.
#' @param y Numeric response (observed only when d = 1).
#' @param z Numeric selection covariates.
#' @param d Integer/logical selection indicator (1 = selected).
#' @return Named list with estimate, se, selection_correction, n, n_selected, method.
#' @keywords internal
hrzs1 <- function(x, y, z, d) {
  y <- as.numeric(y); d <- as.numeric(d)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  Z <- if (is.null(dim(z))) matrix(z, ncol = 1) else as.matrix(z)
  n <- length(y)
  if (n < 20 || nrow(X) != n || nrow(Z) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "sample-selection (insufficient data)"))
  Zc <- if (all(Z[, 1] == 1)) Z else cbind(1, Z)
  gamma <- .hrz_probit_newton(d, Zc)
  eta <- as.numeric(Zc %*% gamma)
  mills <- stats::dnorm(eta) / pmax(stats::pnorm(eta), 1e-8)
  Xc <- if (all(X[, 1] == 1)) X else cbind(1, X)
  sel <- d > 0.5
  if (sum(sel) < max(10, ncol(Xc) + 2))
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "sample-selection (too few selected)"))
  M <- cbind(Xc[sel, , drop = FALSE], mills[sel])
  yy <- y[sel]
  coef <- as.numeric(MASS::ginv(t(M) %*% M) %*% (t(M) %*% yy))
  beta <- coef[seq_len(ncol(Xc))]
  rho_sigma <- coef[length(coef)]
  resid <- yy - as.numeric(M %*% coef)
  sigma2 <- mean(resid^2)
  cov_m <- sigma2 * MASS::ginv(t(M) %*% M)
  se_all <- sqrt(pmax(diag(cov_m), 0))
  list(estimate = as.numeric(beta),
       se = as.numeric(se_all[seq_len(ncol(Xc))]),
       selection_correction = as.numeric(rho_sigma), n = n,
       n_selected = as.integer(sum(sel)),
       method = "Semiparametric Heckman/Powell-Newey-Vella sample selection")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzs1
#' @keywords internal
#' @export
horowitz_sample_selection <- hrzs1
